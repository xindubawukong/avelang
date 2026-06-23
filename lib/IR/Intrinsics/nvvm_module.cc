#include "AST/ast_nodes_expr.h"
#include "Dialect/AveLang/IR/AveLangOps.h"
#include "IR/builtin_module.h"
#include "IR/generator_context.h"
#include "IR/mlir_generator_impl.h"
#include "IR/named_module.h"
#include "Utils/assert.h"
#include "Utils/embedded_filesystem_view.h"
#include "intrinsic_support.h"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/GPU/IR/GPUDialect.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/LLVMIR/NVVMDialect.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/NVGPU/IR/NVGPUDialect.h>
#include <mlir/Dialect/Vector/IR/VectorOps.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/raw_ostream.h>
#include <string>
#include <string_view>
#include <utility>

extern "C" const unsigned char _binary_nvvm_intrinsics_mlirbc_start[];
extern "C" const unsigned char _binary_nvvm_intrinsics_mlirbc_end[];

namespace causalflow::avelang::ir {

using namespace mlir;
using namespace mlir::NVVM;
using namespace causalflow::avelang::dialect;
namespace cf = causalflow::avelang::dialect;

namespace {

enum class CpAsyncBulkIntrinsicKind {
    CommitGroup,
    GlobalSharedCTA,
    Prefetch,
    SharedClusterGlobal,
    SharedClusterSharedCTA,
    TensorGlobalSharedCTA,
    TensorPrefetch,
    TensorReduce,
    TensorSharedClusterGlobal,
    WaitGroup,
};

static llvm::StringRef GetNvvmIntrinsicLibrary() {
    auto *start =
        reinterpret_cast<const char *>(_binary_nvvm_intrinsics_mlirbc_start);
    auto *end =
        reinterpret_cast<const char *>(_binary_nvvm_intrinsics_mlirbc_end);
    return {start, static_cast<size_t>(end - start)};
}

constexpr llvm::StringRef kNvvmIntrinsicLibraryName = "nvvm_intrinsics.mlirbc";
constexpr llvm::StringRef kNvvmIntrinsicLibraryTag =
    "embedded:nvvm_intrinsics.mlirbc";

static void emitInlinePtxVoid(mlir::OpBuilder &builder, mlir::Location loc,
                              llvm::StringRef asmString) {
    mlir::LLVM::InlineAsmOp::create(
        builder, loc, mlir::TypeRange{}, mlir::ValueRange{}, asmString, "",
        /*hasSideEffects=*/true, /*isAlignStack=*/false,
        mlir::LLVM::tailcallkind::TailCallKind::None,
        mlir::LLVM::AsmDialectAttr{}, mlir::ArrayAttr{});
}

static std::optional<int64_t> getConstantIntValue(mlir::Value value) {
    if (!value) {
        return std::nullopt;
    }
    if (auto constOp = value.getDefiningOp<mlir::arith::ConstantOp>()) {
        if (auto intAttr =
                mlir::dyn_cast<mlir::IntegerAttr>(constOp.getValue())) {
            return intAttr.getInt();
        }
    }
    if (auto constOp = value.getDefiningOp<mlir::LLVM::ConstantOp>()) {
        if (auto intAttr =
                mlir::dyn_cast<mlir::IntegerAttr>(constOp.getValue())) {
            return intAttr.getInt();
        }
    }
    return std::nullopt;
}

static bool extractConstantTupleValues(mlir::Value tupleValue,
                                       llvm::SmallVectorImpl<int64_t> &values) {
    if (auto tupleOp = tupleValue.getDefiningOp<cf::MakeIntTupleOp>()) {
        for (auto elem : tupleOp.getElements()) {
            if (!extractConstantTupleValues(elem, values)) {
                return false;
            }
        }
        return true;
    }

    auto value = getConstantIntValue(tupleValue);
    if (!value) {
        return false;
    }
    values.push_back(*value);
    return true;
}

static bool extractTupleValues(mlir::Value tupleValue,
                               llvm::SmallVectorImpl<mlir::Value> &values) {
    if (auto tupleOp = tupleValue.getDefiningOp<cf::MakeIntTupleOp>()) {
        for (auto elem : tupleOp.getElements()) {
            if (!extractTupleValues(elem, values)) {
                return false;
            }
        }
        return true;
    }

    if (!tupleValue) {
        return false;
    }
    values.push_back(tupleValue);
    return true;
}

static bool isMemRefLike(mlir::Type type) {
    return mlir::isa<cf::MemRefType, mlir::MemRefType>(type);
}

static mlir::Value createDefaultIndex(mlir::OpBuilder &builder,
                                      mlir::Location location) {
    return mlir::arith::ConstantIndexOp::create(builder, location, 0)
        .getResult();
}

static mlir::Value castIntegerTo(mlir::OpBuilder &builder,
                                 mlir::Location location, mlir::Value value,
                                 mlir::IntegerType targetType) {
    if (!value) {
        return {};
    }
    if (value.getType() == targetType) {
        return value;
    }
    if (value.getType().isIndex()) {
        return mlir::arith::IndexCastOp::create(builder, location, targetType,
                                                value)
            .getResult();
    }
    auto intType = mlir::dyn_cast<mlir::IntegerType>(value.getType());
    if (!intType) {
        return {};
    }
    if (intType.getWidth() == targetType.getWidth()) {
        return value;
    }
    if (intType.getWidth() < targetType.getWidth()) {
        return mlir::arith::ExtUIOp::create(builder, location, targetType,
                                            value)
            .getResult();
    }
    return mlir::arith::TruncIOp::create(builder, location, targetType, value)
        .getResult();
}

static mlir::Value castToIndex(mlir::OpBuilder &builder,
                               mlir::Location location, mlir::Value value) {
    if (!value) {
        return {};
    }
    if (value.getType().isIndex()) {
        return value;
    }
    if (!value.getType().isIntOrIndex()) {
        return {};
    }
    return mlir::arith::IndexCastOp::create(builder, location,
                                            builder.getIndexType(), value)
        .getResult();
}

static mlir::LLVM::LLVMPointerType
getNvvmPointerType(mlir::OpBuilder &builder,
                   mlir::NVVM::NVVMMemorySpace memorySpace) {
    return mlir::LLVM::LLVMPointerType::get(builder.getContext(),
                                            static_cast<unsigned>(memorySpace));
}

static mlir::Value castToPointer(mlir::OpBuilder &builder,
                                 mlir::Location location, mlir::Value value,
                                 mlir::LLVM::LLVMPointerType pointerType) {
    if (!value) {
        return {};
    }
    if (value.getType() == pointerType) {
        return value;
    }
    auto cast = mlir::UnrealizedConversionCastOp::create(builder, location,
                                                         pointerType, value);
    return cast.getResult(0);
}

static mlir::Value
createPointerFromMemRef(mlir::OpBuilder &builder, mlir::Location location,
                        mlir::Value memref, mlir::Value offsetBytes,
                        mlir::NVVM::NVVMMemorySpace memorySpace) {
    if (!memref) {
        return {};
    }
    auto pointerType = getNvvmPointerType(builder, memorySpace);
    if (mlir::isa<mlir::LLVM::LLVMPointerType>(memref.getType())) {
        return castToPointer(builder, location, memref, pointerType);
    }
    if (!isMemRefLike(memref.getType())) {
        return {};
    }

    auto base = cf::AveLangMemRefExtractAlignedPointerAsIndexOp::create(
        builder, location, builder.getIndexType(), memref);
    mlir::Value addr = base.getResult();
    if (offsetBytes) {
        auto offset = castToIndex(builder, location, offsetBytes);
        if (!offset) {
            return {};
        }
        addr = mlir::arith::AddIOp::create(builder, location, addr, offset);
    }
    auto addrI64 = mlir::arith::IndexCastOp::create(builder, location,
                                                    builder.getI64Type(), addr);
    return mlir::LLVM::IntToPtrOp::create(builder, location, pointerType,
                                          addrI64.getResult())
        .getResult();
}

static mlir::Value createSharedBarrierPointer(mlir::OpBuilder &builder,
                                              mlir::Location location,
                                              mlir::Value value,
                                              mlir::Value offsetBytes) {
    if (!value) {
        return {};
    }
    auto pointerType =
        getNvvmPointerType(builder, mlir::NVVM::NVVMMemorySpace::Shared);
    if (isMemRefLike(value.getType()) ||
        mlir::isa<mlir::LLVM::LLVMPointerType>(value.getType())) {
        return createPointerFromMemRef(builder, location, value, offsetBytes,
                                       mlir::NVVM::NVVMMemorySpace::Shared);
    }
    return castToPointer(builder, location, value, pointerType);
}

static mlir::Value createDescriptorPointer(mlir::OpBuilder &builder,
                                           mlir::Location location,
                                           mlir::Value descriptor) {
    return castToPointer(
        builder, location, descriptor,
        mlir::LLVM::LLVMPointerType::get(builder.getContext()));
}

} // namespace

// NVVM Intrinsics Module
class NVVMIntrinsic : public NamedModule {
  public:
    explicit NVVMIntrinsic();

    void Initialize() override;
    void DeclareModules(mlir::ModuleOp module) override;

    struct MMAConfig {
        int m, n, k;
        MMATypes type_a, type_b;
        MMALayout layout_a, layout_b;
        int fragments_a_count, fragments_b_count, fragments_c_count;
        int fragment_size;
    };

    mlir::Value CreateMma16x8x16F16F16Function(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args) const;
    mlir::Value CreateMma16x8x8F16F32Function(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args) const;

    mlir::Value CreateWgmmaFenceAlignedFunction(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args) const;

    mlir::Value CreateWgmmaGroupSyncAlignedFunction(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args) const;

    mlir::Value CreateWgmmaWaitGroupSyncFunction(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args) const;

    mlir::Value CreateMakeWGMMADescriptorFunction(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args) const;

    mlir::Value CreateMakeTMADescriptorFunction(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args) const;

    mlir::Value
    CreateTMAFenceFunction(ast::Call *call_expr, GeneratorContext *ctx,
                           llvm::ArrayRef<mlir::Value> resolved_args) const;

    mlir::Value
    CreateTMALoadFunction(ast::Call *call_expr, GeneratorContext *ctx,
                          llvm::ArrayRef<mlir::Value> resolved_args) const;

    mlir::Value
    CreateTMAStoreFunction(ast::Call *call_expr, GeneratorContext *ctx,
                           llvm::ArrayRef<mlir::Value> resolved_args) const;

    mlir::Value
    CreateWgmmaAsyncFunction(ast::Call *call_expr, GeneratorContext *ctx,
                             llvm::ArrayRef<mlir::Value> resolved_args) const;

    mlir::Value CreateWgmmaInitAccumulatorFunction(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args) const;

    mlir::Value
    CreateWgmmaStoreFunction(ast::Call *call_expr, GeneratorContext *ctx,
                             llvm::ArrayRef<mlir::Value> resolved_args) const;

    mlir::Value CreateMBarrierCreateFunction(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args) const;

    mlir::Value
    CreateMBarrierInitFunction(ast::Call *call_expr, GeneratorContext *ctx,
                               llvm::ArrayRef<mlir::Value> resolved_args) const;

    mlir::Value CreateMBarrierTryWaitParityFunction(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args) const;

    mlir::Value CreateMBarrierArriveFunction(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args) const;

    mlir::Value CreateMBarrierTestWaitFunction(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args) const;

    mlir::Value CreateMBarrierArriveExpectTxFunction(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args) const;

    mlir::Value CreateCpAsyncCaSharedGlobalFunction(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args) const;

    mlir::Value CreateCpAsyncCommitGroupFunction(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args) const;

    mlir::Value CreateCpAsyncWaitGroupFunction(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args) const;

    mlir::Value
    CreateCpAsyncBulkFunction(ast::Call *call_expr, GeneratorContext *ctx,
                              llvm::ArrayRef<mlir::Value> resolved_args,
                              CpAsyncBulkIntrinsicKind kind) const;

  private:
    void AddLdMatrixFactory(const std::string &name, const std::string &shape,
                            int num, int bit_width, bool transpose);

    void AddStMatrixFactory(const std::string &name, const std::string &shape,
                            int num, int bit_width, bool transpose);

    mlir::Value
    CreateGenericMMAFunction(ast::Call *call_expr, GeneratorContext *ctx,
                             llvm::ArrayRef<mlir::Value> resolved_args,
                             const MMAConfig &config) const;

    mlir::Value
    CreateLdMatrixWithShape(ast::Call *call_expr, GeneratorContext *ctx,
                            llvm::ArrayRef<mlir::Value> resolved_args,
                            const std::string &shape, int num, int bit_width,
                            bool transpose = false) const;

    mlir::Value
    CreateStMatrixWithShape(ast::Call *call_expr, GeneratorContext *ctx,
                            llvm::ArrayRef<mlir::Value> resolved_args,
                            const std::string &shape, int num, int bit_width,
                            bool transpose = false) const;

    bool
    CheckGenericMMAFunction(ast::Call *call_expr, GeneratorContext *ctx,
                            llvm::ArrayRef<mlir::Value> resolved_args) const;

    bool CheckLdMatrixWithShape(ast::Call *call_expr, GeneratorContext *ctx,
                                llvm::ArrayRef<mlir::Value> resolved_args,
                                const std::string &shape, int num,
                                int bit_width, bool transpose) const;

    bool CheckStMatrixWithShape(ast::Call *call_expr, GeneratorContext *ctx,
                                llvm::ArrayRef<mlir::Value> resolved_args,
                                const std::string &shape, int num,
                                int bit_width, bool transpose) const;
    bool CheckWgmmaFenceAlignedFunction(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args) const;

    bool CheckWgmmaGroupSyncAlignedFunction(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args) const;

    bool CheckWgmmaWaitGroupSyncFunction(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args) const;

    bool CheckMakeWGMMADescriptorFunction(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args) const;

    bool
    CheckWgmmaAsyncFunction(ast::Call *call_expr, GeneratorContext *ctx,
                            llvm::ArrayRef<mlir::Value> resolved_args) const;

    bool CheckWgmmaInitAccumulatorFunction(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args) const;

    bool
    CheckWgmmaStoreFunction(ast::Call *call_expr, GeneratorContext *ctx,
                            llvm::ArrayRef<mlir::Value> resolved_args) const;

    bool CheckMBarrierCreateFunction(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args) const;

    bool
    CheckMBarrierInitFunction(ast::Call *call_expr, GeneratorContext *ctx,
                              llvm::ArrayRef<mlir::Value> resolved_args) const;

    bool CheckMBarrierTryWaitParityFunction(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args) const;

    bool CheckMBarrierArriveFunction(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args) const;

    bool CheckMBarrierTestWaitFunction(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args) const;

    bool CheckMBarrierArriveExpectTxFunction(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args) const;

    bool CheckCpAsyncCaSharedGlobalFunction(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args) const;

    bool CheckCpAsyncCommitGroupFunction(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args) const;

    bool CheckCpAsyncWaitGroupFunction(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args) const;

    bool CheckCpAsyncBulkFunction(ast::Call *call_expr, GeneratorContext *ctx,
                                  llvm::ArrayRef<mlir::Value> resolved_args,
                                  CpAsyncBulkIntrinsicKind kind) const;

    bool CheckMakeTMADescriptorFunction(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args) const;

    bool CheckTMAFenceFunction(ast::Call *call_expr, GeneratorContext *ctx,
                               llvm::ArrayRef<mlir::Value> resolved_args) const;

    bool CheckTMALoadFunction(ast::Call *call_expr, GeneratorContext *ctx,
                              llvm::ArrayRef<mlir::Value> resolved_args) const;

    bool CheckTMAStoreFunction(ast::Call *call_expr, GeneratorContext *ctx,
                               llvm::ArrayRef<mlir::Value> resolved_args) const;
};

NVVMIntrinsic::NVVMIntrinsic() : NamedModule("nvvm") {}

void NVVMIntrinsic::Initialize() {
    // Add MMA sync intrinsic function helpers with constraints encoded in
    // function names
    AddFunction(
        "mma_16x8x16_f16_f16",
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> mlir::Value {
            return CreateMma16x8x16F16F16Function(call_expr, gen_ctx,
                                                  resolved_args);
        },
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> bool {
            return CheckGenericMMAFunction(call_expr, gen_ctx, resolved_args);
        });

    AddFunction(
        "mma_16x8x8_f16_f32",
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> mlir::Value {
            return CreateMma16x8x8F16F32Function(call_expr, gen_ctx,
                                                 resolved_args);
        },
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> bool {
            return CheckGenericMMAFunction(call_expr, gen_ctx, resolved_args);
        });

    // LLVM 22 models ldmatrix/stmatrix shape and element type explicitly.
    // The implemented wrappers cover the standard m8n8/b16 family; the old
    // b8 and m16n16 registrations do not match the stricter NVVM verifier.
    for (int num : {1, 2, 4}) {
        std::string base_name =
            "ldmatrix_m8n8_x" + std::to_string(num) + "_b16";
        AddLdMatrixFactory(base_name, "m8n8", num, 16, false);
        AddLdMatrixFactory(base_name + "_trans", "m8n8", num, 16, true);
    }

    for (int num : {1, 2, 4}) {
        std::string base_name =
            "stmatrix_m8n8_x" + std::to_string(num) + "_b16";
        AddStMatrixFactory(base_name, "m8n8", num, 16, false);
        AddStMatrixFactory(base_name + "_trans", "m8n8", num, 16, true);
    }

    AddFunction(
        "wgmma_fence_aligned",
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> mlir::Value {
            return CreateWgmmaFenceAlignedFunction(call_expr, gen_ctx,
                                                   resolved_args);
        },
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> bool {
            return CheckWgmmaFenceAlignedFunction(call_expr, gen_ctx,
                                                  resolved_args);
        });

    AddFunction(
        "wgmma_group_sync_aligned",
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> mlir::Value {
            return CreateWgmmaGroupSyncAlignedFunction(call_expr, gen_ctx,
                                                       resolved_args);
        },
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> bool {
            return CheckWgmmaGroupSyncAlignedFunction(call_expr, gen_ctx,
                                                      resolved_args);
        });

    AddFunction(
        "wgmma_wait_group_sync",
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> mlir::Value {
            return CreateWgmmaWaitGroupSyncFunction(call_expr, gen_ctx,
                                                    resolved_args);
        },
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> bool {
            return CheckWgmmaWaitGroupSyncFunction(call_expr, gen_ctx,
                                                   resolved_args);
        });

    AddFunction(
        "make_wgmma_descriptor",
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> mlir::Value {
            return CreateMakeWGMMADescriptorFunction(call_expr, gen_ctx,
                                                     resolved_args);
        },
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> bool {
            return CheckMakeWGMMADescriptorFunction(call_expr, gen_ctx,
                                                    resolved_args);
        });

    AddFunction(
        "wgmma_async",
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> mlir::Value {
            return CreateWgmmaAsyncFunction(call_expr, gen_ctx, resolved_args);
        },
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> bool {
            return CheckWgmmaAsyncFunction(call_expr, gen_ctx, resolved_args);
        });

    AddFunction(
        "wgmma_init_accumulator",
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> mlir::Value {
            return CreateWgmmaInitAccumulatorFunction(call_expr, gen_ctx,
                                                      resolved_args);
        },
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> bool {
            return CheckWgmmaInitAccumulatorFunction(call_expr, gen_ctx,
                                                     resolved_args);
        });

    AddFunction(
        "wgmma_store",
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> mlir::Value {
            return CreateWgmmaStoreFunction(call_expr, gen_ctx, resolved_args);
        },
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> bool {
            return CheckWgmmaStoreFunction(call_expr, gen_ctx, resolved_args);
        });

    AddFunction(
        "mbarrier_create",
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> mlir::Value {
            return CreateMBarrierCreateFunction(call_expr, gen_ctx,
                                                resolved_args);
        },
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> bool {
            return CheckMBarrierCreateFunction(call_expr, gen_ctx,
                                               resolved_args);
        });

    AddFunction(
        "mbarrier_init",
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> mlir::Value {
            return CreateMBarrierInitFunction(call_expr, gen_ctx,
                                              resolved_args);
        },
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> bool {
            return CheckMBarrierInitFunction(call_expr, gen_ctx, resolved_args);
        });

    AddFunction(
        "mbarrier_try_wait_parity",
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> mlir::Value {
            return CreateMBarrierTryWaitParityFunction(call_expr, gen_ctx,
                                                       resolved_args);
        },
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> bool {
            return CheckMBarrierTryWaitParityFunction(call_expr, gen_ctx,
                                                      resolved_args);
        });

    AddFunction(
        "mbarrier_arrive",
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> mlir::Value {
            return CreateMBarrierArriveFunction(call_expr, gen_ctx,
                                                resolved_args);
        },
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> bool {
            return CheckMBarrierArriveFunction(call_expr, gen_ctx,
                                               resolved_args);
        });

    AddFunction(
        "mbarrier_test_wait",
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> mlir::Value {
            return CreateMBarrierTestWaitFunction(call_expr, gen_ctx,
                                                  resolved_args);
        },
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> bool {
            return CheckMBarrierTestWaitFunction(call_expr, gen_ctx,
                                                 resolved_args);
        });

    AddFunction(
        "mbarrier_arrive_expect_tx",
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> mlir::Value {
            return CreateMBarrierArriveExpectTxFunction(call_expr, gen_ctx,
                                                        resolved_args);
        },
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> bool {
            return CheckMBarrierArriveExpectTxFunction(call_expr, gen_ctx,
                                                       resolved_args);
        });

    AddFunction(
        "cp_async_ca_shared_global",
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> mlir::Value {
            return CreateCpAsyncCaSharedGlobalFunction(call_expr, gen_ctx,
                                                       resolved_args);
        },
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> bool {
            return CheckCpAsyncCaSharedGlobalFunction(call_expr, gen_ctx,
                                                      resolved_args);
        });

    AddFunction(
        "cp_async_commit_group",
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> mlir::Value {
            return CreateCpAsyncCommitGroupFunction(call_expr, gen_ctx,
                                                    resolved_args);
        },
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> bool {
            return CheckCpAsyncCommitGroupFunction(call_expr, gen_ctx,
                                                   resolved_args);
        });

    AddFunction(
        "cp_async_wait_group",
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> mlir::Value {
            return CreateCpAsyncWaitGroupFunction(call_expr, gen_ctx,
                                                  resolved_args);
        },
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> bool {
            return CheckCpAsyncWaitGroupFunction(call_expr, gen_ctx,
                                                 resolved_args);
        });

    auto addCpAsyncBulkFunction = [this](llvm::StringRef name,
                                         CpAsyncBulkIntrinsicKind kind) {
        AddFunction(
            name.str(),
            [this,
             kind](ast::Call *call_expr, GeneratorContext *gen_ctx,
                   llvm::ArrayRef<mlir::Value> resolved_args) -> mlir::Value {
                return CreateCpAsyncBulkFunction(call_expr, gen_ctx,
                                                 resolved_args, kind);
            },
            [this, kind](ast::Call *call_expr, GeneratorContext *gen_ctx,
                         llvm::ArrayRef<mlir::Value> resolved_args) -> bool {
                return CheckCpAsyncBulkFunction(call_expr, gen_ctx,
                                                resolved_args, kind);
            });
    };

    addCpAsyncBulkFunction("cp_async_bulk_commit_group",
                           CpAsyncBulkIntrinsicKind::CommitGroup);
    addCpAsyncBulkFunction("cp_async_bulk_global_shared_cta",
                           CpAsyncBulkIntrinsicKind::GlobalSharedCTA);
    addCpAsyncBulkFunction("cp_async_bulk_prefetch",
                           CpAsyncBulkIntrinsicKind::Prefetch);
    addCpAsyncBulkFunction("cp_async_bulk_shared_cluster_global",
                           CpAsyncBulkIntrinsicKind::SharedClusterGlobal);
    addCpAsyncBulkFunction("cp_async_bulk_shared_cluster_shared_cta",
                           CpAsyncBulkIntrinsicKind::SharedClusterSharedCTA);
    addCpAsyncBulkFunction("cp_async_bulk_tensor_global_shared_cta",
                           CpAsyncBulkIntrinsicKind::TensorGlobalSharedCTA);
    addCpAsyncBulkFunction("cp_async_bulk_tensor_prefetch",
                           CpAsyncBulkIntrinsicKind::TensorPrefetch);
    addCpAsyncBulkFunction("cp_async_bulk_tensor_reduce",
                           CpAsyncBulkIntrinsicKind::TensorReduce);
    addCpAsyncBulkFunction("cp_async_bulk_tensor_shared_cluster_global",
                           CpAsyncBulkIntrinsicKind::TensorSharedClusterGlobal);
    addCpAsyncBulkFunction("cp_async_bulk_wait_group",
                           CpAsyncBulkIntrinsicKind::WaitGroup);

    AddFunction(
        "make_tma_descriptor",
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> mlir::Value {
            return CreateMakeTMADescriptorFunction(call_expr, gen_ctx,
                                                   resolved_args);
        },
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> bool {
            return CheckMakeTMADescriptorFunction(call_expr, gen_ctx,
                                                  resolved_args);
        });

    AddFunction(
        "tma_fence",
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> mlir::Value {
            return CreateTMAFenceFunction(call_expr, gen_ctx, resolved_args);
        },
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> bool {
            return CheckTMAFenceFunction(call_expr, gen_ctx, resolved_args);
        });

    AddFunction(
        "tma_load",
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> mlir::Value {
            return CreateTMALoadFunction(call_expr, gen_ctx, resolved_args);
        },
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> bool {
            return CheckTMALoadFunction(call_expr, gen_ctx, resolved_args);
        });

    AddFunction(
        "tma_store",
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> mlir::Value {
            return CreateTMAStoreFunction(call_expr, gen_ctx, resolved_args);
        },
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> bool {
            return CheckTMAStoreFunction(call_expr, gen_ctx, resolved_args);
        });
}

void NVVMIntrinsic::DeclareModules(mlir::ModuleOp module) {
    if (!module)
        return;

    // Register the bytecode in the intrinsic registry
    auto libraryBytes = GetNvvmIntrinsicLibrary();
    auto &registry = utils::EmbeddedFilesystemView::getInstance();
    registry.registerFile(std::string(kNvvmIntrinsicLibraryName), libraryBytes);

    intrinsics::GetOrCreateImplementationContainer(module, "nvvm",
                                                   kNvvmIntrinsicLibraryTag);

    auto loadDialects = [](mlir::MLIRContext *ctx) {
        ctx->loadDialect<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                         mlir::LLVM::LLVMDialect, mlir::NVVM::NVVMDialect,
                         mlir::memref::MemRefDialect,
                         mlir::vector::VectorDialect, cf::AveLangDialect>();
    };

    if (failed(intrinsics::EnsureIntrinsicDeclarations(
            module, kNvvmIntrinsicLibraryName, libraryBytes, loadDialects))) {
        module.emitError() << "failed to declare NVVM intrinsics";
    }
}

void NVVMIntrinsic::AddLdMatrixFactory(const std::string &name,
                                       const std::string &shape, int num,
                                       int bit_width, bool transpose) {
    AddFunction(
        name,
        [this, shape, num, bit_width,
         transpose](ast::Call *call_expr, GeneratorContext *gen_ctx,
                    llvm::ArrayRef<mlir::Value> resolved_args) -> mlir::Value {
            return CreateLdMatrixWithShape(call_expr, gen_ctx, resolved_args,
                                           shape, num, bit_width, transpose);
        },
        [this, shape, num, bit_width,
         transpose](ast::Call *call_expr, GeneratorContext *gen_ctx,
                    llvm::ArrayRef<mlir::Value> resolved_args) -> bool {
            return CheckLdMatrixWithShape(call_expr, gen_ctx, resolved_args,
                                          shape, num, bit_width, transpose);
        });
}

void NVVMIntrinsic::AddStMatrixFactory(const std::string &name,
                                       const std::string &shape, int num,
                                       int bit_width, bool transpose) {
    AddFunction(
        name,
        [this, shape, num, bit_width,
         transpose](ast::Call *call_expr, GeneratorContext *gen_ctx,
                    llvm::ArrayRef<mlir::Value> resolved_args) -> mlir::Value {
            return CreateStMatrixWithShape(call_expr, gen_ctx, resolved_args,
                                           shape, num, bit_width, transpose);
        },
        [this, shape, num, bit_width,
         transpose](ast::Call *call_expr, GeneratorContext *gen_ctx,
                    llvm::ArrayRef<mlir::Value> resolved_args) -> bool {
            return CheckStMatrixWithShape(call_expr, gen_ctx, resolved_args,
                                          shape, num, bit_width, transpose);
        });
}

mlir::Value NVVMIntrinsic::CreateGenericMMAFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args, const MMAConfig &config) const {
    auto &builder = ctx->GetCurrentFunctionGenerator()->GetBuilder();
    auto location = builder.getUnknownLoc();

    auto a = resolved_args[0];
    auto b = resolved_args[1];
    auto c = resolved_args[2];

    auto c_vector = mlir::dyn_cast<mlir::VectorType>(c.getType());
    if (!c_vector) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "C operand for MMA operation must be a vector type";
        return nullptr;
    }

    // Determine result type based on configuration
    mlir::Type result_type;
    if (c_vector.getElementType().isF16()) {
        result_type = c_vector;
    } else if (c_vector.getElementType().isF32()) {
        result_type = c_vector;
    } else {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "Unsupported result type for MMA operation";
        return nullptr;
    }

    // Create GPUOp NVVM MMA operation
    auto mma_op =
        cf::NVVMMMAOp::create(builder, location, result_type, a, b, c);

    return mma_op.getResult();
}

mlir::Value NVVMIntrinsic::CreateMma16x8x16F16F16Function(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    MMAConfig config = {.m = 16,
                        .n = 8,
                        .k = 16,
                        .type_a = MMATypes::f16,
                        .type_b = MMATypes::f16,
                        .layout_a = MMALayout::row,
                        .layout_b = MMALayout::col,
                        .fragments_a_count = 4,
                        .fragments_b_count = 2,
                        .fragments_c_count = 2,
                        .fragment_size = 2};
    return CreateGenericMMAFunction(call_expr, ctx, resolved_args, config);
}

mlir::Value NVVMIntrinsic::CreateMma16x8x8F16F32Function(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    MMAConfig config = {.m = 16,
                        .n = 8,
                        .k = 8,
                        .type_a = MMATypes::f16,
                        .type_b = MMATypes::f32,
                        .layout_a = MMALayout::row,
                        .layout_b = MMALayout::col,
                        .fragments_a_count = 2,
                        .fragments_b_count = 1,
                        .fragments_c_count = 4,
                        .fragment_size = 2};
    return CreateGenericMMAFunction(call_expr, ctx, resolved_args, config);
}

mlir::Value NVVMIntrinsic::CreateLdMatrixWithShape(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args, const std::string &shape,
    int num, int bit_width, bool transpose) const {
    auto &builder = ctx->GetCurrentFunctionGenerator()->GetBuilder();
    auto location = builder.getUnknownLoc();

    auto memref_ptr = resolved_args[0];

    auto memref_type = mlir::dyn_cast<cf::MemRefType>(memref_ptr.getType());
    if (!memref_type) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "ldmatrix expects a memref argument";
        return nullptr;
    }

    auto is_compatible_memref = [&](const std::string &shape_tag) {
        auto gpu_space = mlir::gpu::AddressSpaceAttr::get(
            builder.getContext(), mlir::gpu::AddressSpace::Workgroup);

        if (memref_type.getMemorySpace() != gpu_space) {
            return false;
        }

        auto elem_type = memref_type.getElementType();

        if (bit_width == 16) {
            bool valid_type = elem_type.isF16() || elem_type.isBF16() ||
                              elem_type.isInteger(16);
            if (!valid_type) {
                return false;
            }
        } else if (bit_width == 8) {
            bool valid_type = elem_type.isInteger(8) ||
                              isa<Float8E4M3FNType>(elem_type) ||
                              isa<Float8E4M3FNUZType>(elem_type);
            if (!valid_type) {
                return false;
            }
        }

        auto shape_vec = memref_type.getShape();
        if (shape_tag == "m8n8") {
            return shape_vec.size() == 2 && shape_vec[0] == 8 &&
                   shape_vec[1] == 8;
        }
        if (shape_tag == "m16n16") {
            return shape_vec.size() == 2 && shape_vec[0] == 16 &&
                   shape_vec[1] == 16;
        }
        return false;
    };

    auto type_to_string = [](mlir::Type type) {
        std::string buffer;
        llvm::raw_string_ostream os(buffer);
        type.print(os);
        return buffer;
    };

    if (!is_compatible_memref(shape)) {
        std::string expected_shape = shape == "m8n8" ? "8x8" : "16x16";
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "ldmatrix_" << shape << " expects memref<" << expected_shape
            << "xb" << bit_width
            << ", #gpu.address_space<workgroup>> (with optional "
               "strides/offset) but found "
            << type_to_string(memref_ptr.getType());
        return nullptr;
    }

    mlir::Type result_type;
    if (num == 1) {
        result_type = builder.getI32Type();
    } else {
        result_type = mlir::VectorType::get({num}, builder.getI32Type());
    }

    auto ld_matrix_op = cf::NVVMLdMatrixOp::create(
        builder, location, result_type, memref_ptr,
        mlir::StringAttr::get(builder.getContext(), shape),
        mlir::IntegerAttr::get(builder.getI32Type(), num),
        mlir::IntegerAttr::get(builder.getI32Type(), bit_width));

    return ld_matrix_op.getResult();
}

mlir::Value NVVMIntrinsic::CreateStMatrixWithShape(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args, const std::string &shape,
    int num, int bit_width, bool transpose) const {
    auto &builder = ctx->GetCurrentFunctionGenerator()->GetBuilder();
    auto location = builder.getUnknownLoc();

    auto memref_ptr = resolved_args[0];

    auto memref_type = mlir::dyn_cast<cf::MemRefType>(memref_ptr.getType());
    if (!memref_type) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "stmatrix expects a memref argument";
        return nullptr;
    }

    auto is_compatible_memref = [&](const std::string &shape_tag) {
        auto gpu_space = mlir::gpu::AddressSpaceAttr::get(
            builder.getContext(), mlir::gpu::AddressSpace::Workgroup);

        if (memref_type.getMemorySpace() != gpu_space) {
            return false;
        }

        auto elem_type = memref_type.getElementType();

        if (bit_width == 16) {
            bool valid_type = elem_type.isF16() || elem_type.isBF16() ||
                              elem_type.isInteger(16);
            if (!valid_type) {
                return false;
            }
        } else if (bit_width == 8) {
            bool valid_type = elem_type.isInteger(8) ||
                              isa<Float8E4M3FNType>(elem_type) ||
                              isa<Float8E4M3FNUZType>(elem_type);
            if (!valid_type) {
                return false;
            }
        }

        auto shape_vec = memref_type.getShape();
        if (shape_tag == "m8n8") {
            return shape_vec.size() == 2 && shape_vec[0] == 8 &&
                   shape_vec[1] == 8;
        }
        if (shape_tag == "m16n16") {
            return shape_vec.size() == 2 && shape_vec[0] == 16 &&
                   shape_vec[1] == 16;
        }
        return false;
    };

    auto type_to_string = [](mlir::Type type) {
        std::string buffer;
        llvm::raw_string_ostream os(buffer);
        type.print(os);
        return buffer;
    };

    if (!is_compatible_memref(shape)) {
        std::string expected_shape = shape == "m8n8" ? "8x8" : "16x16";
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "stmatrix_" << shape << " expects memref<" << expected_shape
            << "xb" << bit_width
            << ", #gpu.address_space<workgroup>> (with optional "
               "strides/offset) but found "
            << type_to_string(memref_ptr.getType());
        return nullptr;
    }

    auto source_arg = resolved_args[1];
    if (!source_arg) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "Failed to generate source operand for stmatrix_" << shape;
        return nullptr;
    }

    auto source_type = source_arg.getType();

    if (num == 1) {
        if (!source_type.isInteger(32) && source_type.isInteger()) {
            source_arg = mlir::arith::ExtSIOp::create(
                builder, location, builder.getI32Type(), source_arg);
            source_type = source_arg.getType();
        }
    } else {
        auto vector_type = mlir::cast<mlir::VectorType>(source_type);
        (void)vector_type;
    }

    cf::NVVMStMatrixOp::create(
        builder, location, source_arg, memref_ptr,
        mlir::StringAttr::get(builder.getContext(), shape),
        mlir::IntegerAttr::get(builder.getI32Type(), num),
        mlir::IntegerAttr::get(builder.getI32Type(), bit_width));

    return ctx->GetCurrentFunctionGenerator()
        ->GetExprGenerator()
        ->CreateVoidValue();
}

bool NVVMIntrinsic::CheckGenericMMAFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    if (resolved_args.size() != 3) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "mma operation requires exactly 3 arguments: a, b, c";
        return false;
    }

    if (!resolved_args[0] || !resolved_args[1] || !resolved_args[2]) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "Failed to generate operands for mma operation";
        return false;
    }

    if (!mlir::dyn_cast<mlir::VectorType>(resolved_args[0].getType()) ||
        !mlir::dyn_cast<mlir::VectorType>(resolved_args[1].getType()) ||
        !mlir::dyn_cast<mlir::VectorType>(resolved_args[2].getType())) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "mma operands must be vector types";
        return false;
    }

    return true;
}

bool NVVMIntrinsic::CheckLdMatrixWithShape(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args, const std::string &shape,
    int num, int bit_width, bool transpose) const {
    if (resolved_args.size() != 1) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "ldmatrix_" << shape << "_x" << num << "_b" << bit_width
            << (transpose ? "_trans" : "")
            << " requires exactly 1 argument: ptr";
        return false;
    }

    if (!resolved_args[0]) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "Failed to generate pointer operand for ldmatrix_" << shape;
        return false;
    }

    if (!mlir::isa<cf::MemRefType>(resolved_args[0].getType())) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "ldmatrix_" << shape << " expects memref pointer operand";
        return false;
    }

    return true;
}

bool NVVMIntrinsic::CheckStMatrixWithShape(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args, const std::string &shape,
    int num, int bit_width, bool transpose) const {
    size_t expected_args = 2;
    if (resolved_args.size() != expected_args) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "stmatrix_" << shape << "_x" << num << "_b" << bit_width
            << (transpose ? "_trans" : "") << " requires exactly "
            << expected_args << " arguments: ptr + vector source values";
        return false;
    }

    if (!resolved_args[0] || !resolved_args[1]) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "Failed to generate operands for stmatrix_" << shape;
        return false;
    }

    if (!mlir::isa<cf::MemRefType>(resolved_args[0].getType())) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "stmatrix_" << shape << " expects memref pointer operand";
        return false;
    }

    auto source_type = resolved_args[1].getType();
    if (num == 1) {
        if (!source_type.isInteger(32)) {
            ctx->diagnostic_manager->Report(
                basic::DiagnosticCode::kUnimplemented,
                call_expr->GetSourceRange().getBegin())
                << "stmatrix_" << shape
                << "_x1 source operand must be integer type";
            return false;
        }
    } else {
        auto vector_type = mlir::dyn_cast<mlir::VectorType>(source_type);
        if (!vector_type || vector_type.getNumElements() != num ||
            !vector_type.getElementType().isInteger(32)) {
            ctx->diagnostic_manager->Report(
                basic::DiagnosticCode::kUnimplemented,
                call_expr->GetSourceRange().getBegin())
                << "stmatrix_" << shape << "_x" << num
                << " source operand must be i32x" << num << " vector type";
            return false;
        }
    }

    return true;
}

mlir::Value NVVMIntrinsic::CreateWgmmaFenceAlignedFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    auto &builder = ctx->GetCurrentFunctionGenerator()->GetBuilder();
    auto location = builder.getUnknownLoc();

    emitInlinePtxVoid(builder, location, "wgmma.fence.sync.aligned;");
    return ctx->GetCurrentFunctionGenerator()
        ->GetExprGenerator()
        ->CreateVoidValue();
}

bool NVVMIntrinsic::CheckWgmmaFenceAlignedFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    if (!resolved_args.empty()) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "wgmma_fence_aligned requires no arguments";
        return false;
    }

    return true;
}

mlir::Value NVVMIntrinsic::CreateWgmmaGroupSyncAlignedFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    auto &builder = ctx->GetCurrentFunctionGenerator()->GetBuilder();
    auto location = builder.getUnknownLoc();

    emitInlinePtxVoid(builder, location, "wgmma.commit_group.sync.aligned;");
    return ctx->GetCurrentFunctionGenerator()
        ->GetExprGenerator()
        ->CreateVoidValue();
}

bool NVVMIntrinsic::CheckWgmmaGroupSyncAlignedFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    if (!resolved_args.empty()) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "wgmma_group_sync_aligned requires no arguments";
        return false;
    }

    return true;
}

mlir::Value NVVMIntrinsic::CreateWgmmaWaitGroupSyncFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    auto &builder = ctx->GetCurrentFunctionGenerator()->GetBuilder();
    auto location = builder.getUnknownLoc();

    if (!CheckWgmmaWaitGroupSyncFunction(call_expr, ctx, resolved_args)) {
        return nullptr;
    }

    auto groupValue = getConstantIntValue(resolved_args[0]);
    if (!groupValue) {
        return nullptr;
    }

    std::string asmString =
        "wgmma.wait_group.sync.aligned " + std::to_string(*groupValue) + ";";
    emitInlinePtxVoid(builder, location, asmString);
    return ctx->GetCurrentFunctionGenerator()
        ->GetExprGenerator()
        ->CreateVoidValue();
}

bool NVVMIntrinsic::CheckWgmmaWaitGroupSyncFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    if (resolved_args.size() != 1) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "wgmma_wait_group_sync requires exactly 1 argument: group";
        return false;
    }

    if (!resolved_args[0]) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "Failed to generate group operand for wgmma_wait_group_sync";
        return false;
    }

    if (!resolved_args[0].getType().isIntOrIndex()) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "wgmma_wait_group_sync group operand must be integer type";
        return false;
    }

    if (!getConstantIntValue(resolved_args[0])) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "wgmma_wait_group_sync requires a constant integer value for "
               "group";
        return false;
    }

    return true;
}

mlir::Value NVVMIntrinsic::CreateMakeWGMMADescriptorFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    auto &builder = ctx->GetCurrentFunctionGenerator()->GetBuilder();
    auto location = builder.getUnknownLoc();

    if (!CheckMakeWGMMADescriptorFunction(call_expr, ctx, resolved_args)) {
        return nullptr;
    }

    auto substrateMemrefType =
        mlir::dyn_cast<cf::MemRefType>(resolved_args[0].getType());
    auto workgroupMemorySpace = mlir::IntegerAttr::get(
        mlir::IntegerType::get(builder.getContext(), 64), 3);
    auto memrefType = mlir::MemRefType::getChecked(
        [&]() { return mlir::emitError(location); },
        substrateMemrefType.getShape(), substrateMemrefType.getElementType(),
        mlir::MemRefLayoutAttrInterface(), workgroupMemorySpace);
    if (!memrefType) {
        return nullptr;
    }

    auto resultType = mlir::nvgpu::WarpgroupMatrixDescriptorType::get(
        builder.getContext(), memrefType);
    auto getI32Attr = [&](mlir::Value value) {
        return mlir::IntegerAttr::get(builder.getI32Type(),
                                      *getConstantIntValue(value));
    };

    auto descriptor = cf::NVVMWGMMADescriptorOp::create(
        builder, location, resultType, resolved_args[0],
        getI32Attr(resolved_args[1]), getI32Attr(resolved_args[2]),
        getI32Attr(resolved_args[3]), getI32Attr(resolved_args[4]));
    return descriptor.getResult();
}

bool NVVMIntrinsic::CheckMakeWGMMADescriptorFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    if (resolved_args.size() != 5) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "make_wgmma_descriptor requires exactly 5 arguments: tensor, "
               "swizzle_kind, l2promo_kind, oob_kind, interleave_kind";
        return false;
    }

    if (!resolved_args[0]) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "Failed to generate tensor operand for make_wgmma_descriptor";
        return false;
    }

    if (!mlir::isa<cf::MemRefType>(resolved_args[0].getType())) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "make_wgmma_descriptor expects memref pointer operand for "
               "tensor";
        return false;
    }

    auto checkKind = [&](size_t index, llvm::StringRef name, int64_t maxValue) {
        if (!resolved_args[index]) {
            ctx->diagnostic_manager->Report(
                basic::DiagnosticCode::kUnimplemented,
                call_expr->GetSourceRange().getBegin())
                << "Failed to generate " << name
                << " operand for make_wgmma_descriptor";
            return false;
        }
        if (!resolved_args[index].getType().isIntOrIndex()) {
            ctx->diagnostic_manager->Report(
                basic::DiagnosticCode::kUnimplemented,
                call_expr->GetSourceRange().getBegin())
                << "make_wgmma_descriptor " << name
                << " operand must be an integer type";
            return false;
        }
        auto value = getConstantIntValue(resolved_args[index]);
        if (!value) {
            ctx->diagnostic_manager->Report(
                basic::DiagnosticCode::kUnimplemented,
                call_expr->GetSourceRange().getBegin())
                << "make_wgmma_descriptor requires a constant integer value "
                   "for "
                << name;
            return false;
        }
        if (*value < 0 || *value > maxValue) {
            ctx->diagnostic_manager->Report(
                basic::DiagnosticCode::kUnimplemented,
                call_expr->GetSourceRange().getBegin())
                << "make_wgmma_descriptor " << name
                << " operand has invalid value " << *value;
            return false;
        }
        return true;
    };

    return checkKind(1, "swizzle_kind", 3) && checkKind(2, "l2promo_kind", 3) &&
           checkKind(3, "oob_kind", 1) && checkKind(4, "interleave_kind", 2);
}

mlir::Value NVVMIntrinsic::CreateWgmmaAsyncFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    auto &builder = ctx->GetCurrentFunctionGenerator()->GetBuilder();
    auto location = builder.getUnknownLoc();

    if (!CheckWgmmaAsyncFunction(call_expr, ctx, resolved_args)) {
        return nullptr;
    }

    auto async = cf::NVVMWGMMAAsyncOp::create(
        builder, location, resolved_args[2].getType(), resolved_args[0],
        resolved_args[1], resolved_args[2]);
    return async.getResult();
}

bool NVVMIntrinsic::CheckWgmmaAsyncFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    if (resolved_args.size() != 3) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "wgmma_async requires exactly 3 arguments: desc_a, desc_b, acc";
        return false;
    }

    if (!resolved_args[0] || !resolved_args[1] || !resolved_args[2]) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "Failed to generate operands for wgmma_async";
        return false;
    }

    if (!mlir::isa<mlir::nvgpu::WarpgroupMatrixDescriptorType>(
            resolved_args[0].getType()) ||
        !mlir::isa<mlir::nvgpu::WarpgroupMatrixDescriptorType>(
            resolved_args[1].getType())) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "wgmma_async descriptor operands must be wgmma_descriptor";
        return false;
    }

    if (!mlir::isa<mlir::nvgpu::WarpgroupAccumulatorType>(
            resolved_args[2].getType())) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "wgmma_async acc operand must be of type warpgroup_accumulator";
        return false;
    }

    return true;
}

mlir::Value NVVMIntrinsic::CreateWgmmaInitAccumulatorFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    auto &builder = ctx->GetCurrentFunctionGenerator()->GetBuilder();
    auto location = builder.getUnknownLoc();

    if (!CheckWgmmaInitAccumulatorFunction(call_expr, ctx, resolved_args)) {
        return nullptr;
    }

    auto mSize = *getConstantIntValue(resolved_args[0]);
    auto nSize = *getConstantIntValue(resolved_args[1]);
    auto vecType = mlir::VectorType::get({mSize, nSize}, builder.getF32Type());
    auto accType = mlir::nvgpu::WarpgroupAccumulatorType::get(
        builder.getContext(), vecType);

    auto acc = mlir::nvgpu::WarpgroupMmaInitAccumulatorOp::create(
        builder, location, accType);
    return acc.getResult();
}

bool NVVMIntrinsic::CheckWgmmaInitAccumulatorFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    if (resolved_args.size() != 2) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "wgmma_init_accumulator requires exactly 2 arguments: m, n";
        return false;
    }

    for (size_t i = 0; i < resolved_args.size(); ++i) {
        llvm::StringRef name = i == 0 ? "m" : "n";
        if (!resolved_args[i]) {
            ctx->diagnostic_manager->Report(
                basic::DiagnosticCode::kUnimplemented,
                call_expr->GetSourceRange().getBegin())
                << "Failed to generate " << name
                << " operand for wgmma_init_accumulator";
            return false;
        }
        if (!resolved_args[i].getType().isIntOrIndex()) {
            ctx->diagnostic_manager->Report(
                basic::DiagnosticCode::kUnimplemented,
                call_expr->GetSourceRange().getBegin())
                << "wgmma_init_accumulator " << name
                << " operand must be an integer type";
            return false;
        }
        if (!getConstantIntValue(resolved_args[i])) {
            ctx->diagnostic_manager->Report(
                basic::DiagnosticCode::kUnimplemented,
                call_expr->GetSourceRange().getBegin())
                << "wgmma_init_accumulator " << name
                << " operand must be a constant integer";
            return false;
        }
    }

    return true;
}

mlir::Value NVVMIntrinsic::CreateWgmmaStoreFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    auto &builder = ctx->GetCurrentFunctionGenerator()->GetBuilder();
    auto location = builder.getUnknownLoc();

    if (!CheckWgmmaStoreFunction(call_expr, ctx, resolved_args)) {
        return nullptr;
    }

    cf::NVVMWGMMAStoreOp::create(builder, location, resolved_args[0],
                                 resolved_args[1]);
    return ctx->GetCurrentFunctionGenerator()
        ->GetExprGenerator()
        ->CreateVoidValue();
}

bool NVVMIntrinsic::CheckWgmmaStoreFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    if (resolved_args.size() != 2) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "wgmma_store requires exactly 2 arguments: acc, dst";
        return false;
    }

    if (!resolved_args[0] || !resolved_args[1]) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "Failed to generate operands for wgmma_store";
        return false;
    }

    if (!mlir::isa<mlir::nvgpu::WarpgroupAccumulatorType>(
            resolved_args[0].getType())) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "wgmma_store acc operand must be of type warpgroup_accumulator";
        return false;
    }

    if (!mlir::isa<cf::MemRefType>(resolved_args[1].getType())) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "wgmma_store dst operand must be of memref type";
        return false;
    }

    return true;
}

mlir::Value NVVMIntrinsic::CreateMBarrierCreateFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    auto &builder = ctx->GetCurrentFunctionGenerator()->GetBuilder();
    auto location = builder.getUnknownLoc();

    if (!CheckMBarrierCreateFunction(call_expr, ctx, resolved_args)) {
        return nullptr;
    }

    auto workgroupSpace = mlir::gpu::AddressSpaceAttr::get(
        builder.getContext(), mlir::gpu::AddressSpace::Workgroup);
    auto barrierType = mlir::nvgpu::MBarrierGroupType::get(builder.getContext(),
                                                           workgroupSpace);
    auto barrier =
        mlir::nvgpu::MBarrierCreateOp::create(builder, location, barrierType);
    return barrier.getResult();
}

bool NVVMIntrinsic::CheckMBarrierCreateFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    if (!resolved_args.empty()) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "mbarrier_create does not take any arguments";
        return false;
    }

    return true;
}

mlir::Value NVVMIntrinsic::CreateMBarrierInitFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    auto &builder = ctx->GetCurrentFunctionGenerator()->GetBuilder();
    auto location = builder.getUnknownLoc();

    if (!CheckMBarrierInitFunction(call_expr, ctx, resolved_args)) {
        return nullptr;
    }

    const auto &keywords = call_expr->GetKeywords();
    size_t keywordCount = keywords.size();
    size_t positionalCount = resolved_args.size() - keywordCount;

    mlir::Value mbarId = positionalCount > 1 ? resolved_args[1] : mlir::Value();
    mlir::Value count =
        positionalCount > 2
            ? resolved_args[2]
            : mlir::arith::ConstantIndexOp::create(builder, location, 0)
                  .getResult();
    mlir::Value predicate =
        positionalCount > 3 ? resolved_args[3] : mlir::Value();

    for (size_t i = 0; i < keywordCount; ++i) {
        mlir::Value value = resolved_args[positionalCount + i];
        llvm::StringRef name = keywords[i];
        if (name == "mbar_id") {
            mbarId = value;
        } else if (name == "count") {
            count = value;
        } else if (name == "predicate") {
            predicate = value;
        }
    }

    auto mbarIdValue = getConstantIntValue(mbarId);
    if (!mbarIdValue) {
        return nullptr;
    }

    if (!count.getType().isIndex()) {
        count = mlir::arith::IndexCastOp::create(builder, location,
                                                 builder.getIndexType(), count);
    }
    if (predicate) {
        if (predicate.getType().isIndex()) {
            predicate = mlir::arith::IndexCastOp::create(
                builder, location, builder.getI1Type(), predicate);
        } else if (auto intType =
                       mlir::dyn_cast<mlir::IntegerType>(predicate.getType());
                   intType && intType.getWidth() != 1) {
            predicate = mlir::arith::TruncIOp::create(
                builder, location, builder.getI1Type(), predicate);
        }
    }

    auto mbarIdIndex =
        mlir::arith::ConstantIndexOp::create(builder, location, *mbarIdValue);
    mlir::nvgpu::MBarrierInitOp::create(builder, location, resolved_args[0],
                                        count, mbarIdIndex, predicate);
    mlir::gpu::BarrierOp::create(builder, location);

    return ctx->GetCurrentFunctionGenerator()
        ->GetExprGenerator()
        ->CreateVoidValue();
}

bool NVVMIntrinsic::CheckMBarrierInitFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    const auto &keywords = call_expr->GetKeywords();
    if (resolved_args.size() < keywords.size()) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "mbarrier_init internal argument mismatch";
        return false;
    }

    size_t keywordCount = keywords.size();
    size_t positionalCount = resolved_args.size() - keywordCount;
    if (positionalCount < 1 || positionalCount > 4) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "mbarrier_init requires barrier, mbar_id and optional count, "
               "predicate";
        return false;
    }

    for (auto name : keywords) {
        if (name != "mbar_id" && name != "count" && name != "predicate") {
            ctx->diagnostic_manager->Report(
                basic::DiagnosticCode::kUnimplemented,
                call_expr->GetSourceRange().getBegin())
                << "mbarrier_init got unsupported keyword argument '" << name
                << "'";
            return false;
        }
    }

    if (!resolved_args[0]) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "Failed to generate barrier operand for mbarrier_init";
        return false;
    }

    if (!mlir::isa<mlir::nvgpu::MBarrierGroupType>(
            resolved_args[0].getType())) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "mbarrier_init operand must be of type mbarrier_group_t";
        return false;
    }

    auto keywordValue = [&](llvm::StringRef wanted) -> mlir::Value {
        for (size_t i = 0; i < keywordCount; ++i) {
            if (keywords[i] == wanted) {
                return resolved_args[positionalCount + i];
            }
        }
        return {};
    };

    mlir::Value mbarId =
        positionalCount > 1 ? resolved_args[1] : keywordValue("mbar_id");
    if (!mbarId) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "Failed to generate mbar_id operand for mbarrier_init";
        return false;
    }

    auto checkIntArg = [&](mlir::Value value, llvm::StringRef name) -> bool {
        if (value && !value.getType().isIntOrIndex()) {
            ctx->diagnostic_manager->Report(
                basic::DiagnosticCode::kUnimplemented,
                call_expr->GetSourceRange().getBegin())
                << "mbarrier_init " << name
                << " operand must be an integer type";
            return false;
        }
        return true;
    };

    if (!checkIntArg(mbarId, "mbar_id") || !getConstantIntValue(mbarId)) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "mbarrier_init requires a constant integer value for mbar_id";
        return false;
    }
    if (positionalCount > 2 && !checkIntArg(resolved_args[2], "count")) {
        return false;
    }
    if (positionalCount > 3 && !checkIntArg(resolved_args[3], "predicate")) {
        return false;
    }
    for (size_t i = 0; i < keywordCount; ++i) {
        if (!checkIntArg(resolved_args[positionalCount + i], keywords[i])) {
            return false;
        }
    }

    return true;
}

mlir::Value NVVMIntrinsic::CreateMBarrierTryWaitParityFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    auto &builder = ctx->GetCurrentFunctionGenerator()->GetBuilder();
    auto location = builder.getUnknownLoc();

    if (!CheckMBarrierTryWaitParityFunction(call_expr, ctx, resolved_args)) {
        return nullptr;
    }

    auto parityValue = *getConstantIntValue(resolved_args[1]);
    auto ticksValue = *getConstantIntValue(resolved_args[2]);
    auto mbarIdValue = *getConstantIntValue(resolved_args[3]);

    auto parity = mlir::LLVM::ConstantOp::create(
        builder, location, builder.getI1Type(), parityValue);
    auto ticks =
        mlir::arith::ConstantIndexOp::create(builder, location, ticksValue);
    auto mbarId =
        mlir::arith::ConstantIndexOp::create(builder, location, mbarIdValue);
    mlir::nvgpu::MBarrierTryWaitParityOp::create(
        builder, location, resolved_args[0], parity, ticks, mbarId);

    return ctx->GetCurrentFunctionGenerator()
        ->GetExprGenerator()
        ->CreateVoidValue();
}

bool NVVMIntrinsic::CheckMBarrierTryWaitParityFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    if (resolved_args.size() != 4) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "mbarrier_try_wait_parity requires exactly 4 arguments: "
               "barrier, phaseParity, ticks, mbarId";
        return false;
    }

    if (!resolved_args[0] || !resolved_args[1] || !resolved_args[2] ||
        !resolved_args[3]) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "Failed to generate operands for mbarrier_try_wait_parity";
        return false;
    }

    if (!mlir::isa<mlir::nvgpu::MBarrierGroupType>(
            resolved_args[0].getType())) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "mbarrier_try_wait_parity operand must be of type "
               "mbarrier_group_t";
        return false;
    }

    for (size_t i = 1; i < resolved_args.size(); ++i) {
        if (!resolved_args[i].getType().isIntOrIndex() ||
            !getConstantIntValue(resolved_args[i])) {
            ctx->diagnostic_manager->Report(
                basic::DiagnosticCode::kUnimplemented,
                call_expr->GetSourceRange().getBegin())
                << "mbarrier_try_wait_parity phaseParity, ticks, and mbarId "
                   "operands must be constant integers";
            return false;
        }
    }

    return true;
}

mlir::Value NVVMIntrinsic::CreateMBarrierArriveFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    auto &builder = ctx->GetCurrentFunctionGenerator()->GetBuilder();
    auto location = builder.getUnknownLoc();

    if (!CheckMBarrierArriveFunction(call_expr, ctx, resolved_args)) {
        return nullptr;
    }

    auto mbarIdValue = *getConstantIntValue(resolved_args[1]);
    auto mbarId =
        mlir::arith::ConstantIndexOp::create(builder, location, mbarIdValue);
    auto token = mlir::nvgpu::MBarrierArriveOp::create(
        builder, location, resolved_args[0], mbarId);
    return token.getResult();
}

bool NVVMIntrinsic::CheckMBarrierArriveFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    if (resolved_args.size() != 2) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "mbarrier_arrive requires exactly 2 arguments: barrier, mbarId";
        return false;
    }

    if (!resolved_args[0] || !resolved_args[1]) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "Failed to generate operands for mbarrier_arrive";
        return false;
    }

    if (!mlir::isa<mlir::nvgpu::MBarrierGroupType>(
            resolved_args[0].getType())) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "mbarrier_arrive operand must be of type mbarrier_group_t";
        return false;
    }

    if (!resolved_args[1].getType().isIntOrIndex() ||
        !getConstantIntValue(resolved_args[1])) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "mbarrier_arrive mbarId operand must be a constant integer";
        return false;
    }

    return true;
}

mlir::Value NVVMIntrinsic::CreateMBarrierTestWaitFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    auto &builder = ctx->GetCurrentFunctionGenerator()->GetBuilder();
    auto location = builder.getUnknownLoc();

    if (!CheckMBarrierTestWaitFunction(call_expr, ctx, resolved_args)) {
        return nullptr;
    }

    auto mbarIdValue = *getConstantIntValue(resolved_args[2]);
    auto mbarId =
        mlir::arith::ConstantIndexOp::create(builder, location, mbarIdValue);
    auto ready = mlir::nvgpu::MBarrierTestWaitOp::create(
        builder, location, resolved_args[0], resolved_args[1], mbarId);
    return ready.getResult();
}

bool NVVMIntrinsic::CheckMBarrierTestWaitFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    if (resolved_args.size() != 3) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "mbarrier_test_wait requires exactly 3 arguments: barrier, "
               "token, mbarId";
        return false;
    }

    if (!resolved_args[0] || !resolved_args[1] || !resolved_args[2]) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "Failed to generate operands for mbarrier_test_wait";
        return false;
    }

    if (!mlir::isa<mlir::nvgpu::MBarrierGroupType>(
            resolved_args[0].getType())) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "mbarrier_test_wait operand must be of type mbarrier_group_t";
        return false;
    }

    if (!mlir::isa<mlir::nvgpu::MBarrierTokenType>(
            resolved_args[1].getType())) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "mbarrier_test_wait token operand must be of type "
               "mbarrier_token_t";
        return false;
    }

    if (!resolved_args[2].getType().isIntOrIndex() ||
        !getConstantIntValue(resolved_args[2])) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "mbarrier_test_wait mbarId operand must be a constant integer";
        return false;
    }

    return true;
}

mlir::Value NVVMIntrinsic::CreateMBarrierArriveExpectTxFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    auto &builder = ctx->GetCurrentFunctionGenerator()->GetBuilder();
    auto location = builder.getUnknownLoc();

    if (!CheckMBarrierArriveExpectTxFunction(call_expr, ctx, resolved_args)) {
        return nullptr;
    }

    auto txCountValue = *getConstantIntValue(resolved_args[1]);
    auto mbarIdValue = *getConstantIntValue(resolved_args[2]);
    auto predicateValue = *getConstantIntValue(resolved_args[3]);

    auto txCount =
        mlir::arith::ConstantIndexOp::create(builder, location, txCountValue);
    auto mbarId =
        mlir::arith::ConstantIndexOp::create(builder, location, mbarIdValue);
    auto predicate = mlir::LLVM::ConstantOp::create(
        builder, location, builder.getI1Type(), predicateValue);
    mlir::nvgpu::MBarrierArriveExpectTxOp::create(
        builder, location, resolved_args[0], txCount, mbarId, predicate);

    return ctx->GetCurrentFunctionGenerator()
        ->GetExprGenerator()
        ->CreateVoidValue();
}

bool NVVMIntrinsic::CheckMBarrierArriveExpectTxFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    if (resolved_args.size() != 4) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "mbarrier_arrive_expect_tx requires exactly 4 arguments: "
               "barrier, txcount, mbarId, predicate";
        return false;
    }

    if (!resolved_args[0] || !resolved_args[1] || !resolved_args[2] ||
        !resolved_args[3]) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "Failed to generate operands for mbarrier_arrive_expect_tx";
        return false;
    }

    if (!mlir::isa<mlir::nvgpu::MBarrierGroupType>(
            resolved_args[0].getType())) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "mbarrier_arrive_expect_tx operand must be of type "
               "mbarrier_group_t";
        return false;
    }

    for (size_t i = 1; i < resolved_args.size(); ++i) {
        if (!resolved_args[i].getType().isIntOrIndex() ||
            !getConstantIntValue(resolved_args[i])) {
            ctx->diagnostic_manager->Report(
                basic::DiagnosticCode::kUnimplemented,
                call_expr->GetSourceRange().getBegin())
                << "mbarrier_arrive_expect_tx txcount, mbarId, and predicate "
                   "operands must be constant integers";
            return false;
        }
    }

    return true;
}

mlir::Value NVVMIntrinsic::CreateCpAsyncCaSharedGlobalFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    auto &builder = ctx->GetCurrentFunctionGenerator()->GetBuilder();
    auto location = builder.getUnknownLoc();

    if (!CheckCpAsyncCaSharedGlobalFunction(call_expr, ctx, resolved_args)) {
        return nullptr;
    }

    auto toIndex = [&](mlir::Value value) -> mlir::Value {
        if (value.getType().isIndex()) {
            return value;
        }
        return mlir::arith::IndexCastOp::create(builder, location,
                                                builder.getIndexType(), value);
    };

    auto dstBase = cf::AveLangMemRefExtractAlignedPointerAsIndexOp::create(
        builder, location, builder.getIndexType(), resolved_args[0]);
    auto srcBase = cf::AveLangMemRefExtractAlignedPointerAsIndexOp::create(
        builder, location, builder.getIndexType(), resolved_args[1]);
    auto dstOffset = toIndex(resolved_args[2]);
    auto srcOffset = toIndex(resolved_args[3]);
    auto dstAddr = mlir::arith::AddIOp::create(builder, location,
                                               dstBase.getResult(), dstOffset);
    auto srcAddr = mlir::arith::AddIOp::create(builder, location,
                                               srcBase.getResult(), srcOffset);

    auto dstAddrI32 = mlir::arith::IndexCastOp::create(
        builder, location, builder.getI32Type(), dstAddr.getResult());
    auto srcAddrI32 = mlir::arith::IndexCastOp::create(
        builder, location, builder.getI32Type(), srcAddr.getResult());
    auto dstAddrI64 = mlir::arith::ExtUIOp::create(
        builder, location, builder.getI64Type(), dstAddrI32.getResult());
    auto srcAddrI64 = mlir::arith::ExtUIOp::create(
        builder, location, builder.getI64Type(), srcAddrI32.getResult());

    auto dstPtrType = mlir::LLVM::LLVMPointerType::get(
        builder.getContext(),
        static_cast<unsigned>(mlir::NVVM::NVVMMemorySpace::Shared));
    auto srcPtrType = mlir::LLVM::LLVMPointerType::get(
        builder.getContext(),
        static_cast<unsigned>(mlir::NVVM::NVVMMemorySpace::Global));

    auto dstPtr = mlir::LLVM::IntToPtrOp::create(builder, location, dstPtrType,
                                                 dstAddrI64.getResult());
    auto srcPtr = mlir::LLVM::IntToPtrOp::create(builder, location, srcPtrType,
                                                 srcAddrI64.getResult());
    auto sizeBytes =
        static_cast<int64_t>(*getConstantIntValue(resolved_args[4]));

    mlir::NVVM::CpAsyncOp::create(
        builder, location, dstPtr.getResult(), srcPtr.getResult(),
        builder.getI32IntegerAttr(sizeBytes),
        mlir::NVVM::LoadCacheModifierKindAttr::get(
            builder.getContext(), mlir::NVVM::LoadCacheModifierKind::CA),
        mlir::Value());

    return ctx->GetCurrentFunctionGenerator()
        ->GetExprGenerator()
        ->CreateVoidValue();
}

bool NVVMIntrinsic::CheckCpAsyncCaSharedGlobalFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    if (resolved_args.size() != 5) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "cp_async_ca_shared_global requires 5 arguments: dst, src, "
               "dst_offset_bytes, src_offset_bytes, size_bytes";
        return false;
    }
    if (!resolved_args[0] || !resolved_args[1] || !resolved_args[2] ||
        !resolved_args[3] || !resolved_args[4]) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "failed to generate one or more operands for "
               "cp_async_ca_shared_global";
        return false;
    }

    auto dstType = mlir::dyn_cast<cf::MemRefType>(resolved_args[0].getType());
    auto srcType = mlir::dyn_cast<cf::MemRefType>(resolved_args[1].getType());
    if (!dstType || !srcType) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "cp_async_ca_shared_global dst/src operands must be memrefs";
        return false;
    }

    auto gpuSpace = mlir::gpu::AddressSpaceAttr::get(
        ctx->GetCurrentFunctionGenerator()->GetBuilder().getContext(),
        mlir::gpu::AddressSpace::Workgroup);
    if (dstType.getMemorySpace() != gpuSpace) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "cp_async_ca_shared_global dst operand must be in workgroup "
               "memory";
        return false;
    }

    if (!resolved_args[2].getType().isIntOrIndex() ||
        !resolved_args[3].getType().isIntOrIndex() ||
        !resolved_args[4].getType().isIntOrIndex()) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "cp_async_ca_shared_global offset/size operands must be "
               "integer/index";
        return false;
    }

    auto sizeBytes = getConstantIntValue(resolved_args[4]);
    if (!sizeBytes) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "cp_async_ca_shared_global size_bytes must be a compile-time "
               "constant";
        return false;
    }

    if (*sizeBytes != 4 && *sizeBytes != 8 && *sizeBytes != 16) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "cp_async_ca_shared_global size_bytes must be one of 4, 8, or "
               "16";
        return false;
    }

    return true;
}

mlir::Value NVVMIntrinsic::CreateCpAsyncCommitGroupFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    auto &builder = ctx->GetCurrentFunctionGenerator()->GetBuilder();
    auto location = builder.getUnknownLoc();

    if (!CheckCpAsyncCommitGroupFunction(call_expr, ctx, resolved_args)) {
        return nullptr;
    }

    mlir::NVVM::CpAsyncCommitGroupOp::create(builder, location);
    return ctx->GetCurrentFunctionGenerator()
        ->GetExprGenerator()
        ->CreateVoidValue();
}

bool NVVMIntrinsic::CheckCpAsyncCommitGroupFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    if (!resolved_args.empty()) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "cp_async_commit_group takes no arguments";
        return false;
    }
    return true;
}

mlir::Value NVVMIntrinsic::CreateCpAsyncWaitGroupFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    auto &builder = ctx->GetCurrentFunctionGenerator()->GetBuilder();
    auto location = builder.getUnknownLoc();

    if (!CheckCpAsyncWaitGroupFunction(call_expr, ctx, resolved_args)) {
        return nullptr;
    }

    auto group = static_cast<int32_t>(*getConstantIntValue(resolved_args[0]));
    mlir::NVVM::CpAsyncWaitGroupOp::create(builder, location,
                                           builder.getI32IntegerAttr(group));
    return ctx->GetCurrentFunctionGenerator()
        ->GetExprGenerator()
        ->CreateVoidValue();
}

bool NVVMIntrinsic::CheckCpAsyncWaitGroupFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    if (resolved_args.size() != 1 || !resolved_args[0]) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "cp_async_wait_group requires 1 argument: n";
        return false;
    }
    if (!resolved_args[0].getType().isIntOrIndex()) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "cp_async_wait_group n operand must be integer/index";
        return false;
    }

    auto group = getConstantIntValue(resolved_args[0]);
    if (!group) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "cp_async_wait_group n must be a compile-time constant";
        return false;
    }
    if (*group < 0 || *group > 8) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "cp_async_wait_group n must be in [0, 8]";
        return false;
    }
    return true;
}

mlir::Value NVVMIntrinsic::CreateCpAsyncBulkFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args,
    CpAsyncBulkIntrinsicKind kind) const {
    auto &builder = ctx->GetCurrentFunctionGenerator()->GetBuilder();
    auto location = builder.getUnknownLoc();

    if (!CheckCpAsyncBulkFunction(call_expr, ctx, resolved_args, kind)) {
        return nullptr;
    }

    const auto &keywords = call_expr->GetKeywords();
    size_t keywordCount = keywords.size();
    size_t positionalCount = resolved_args.size() - keywordCount;

    auto keywordValue = [&](llvm::StringRef key) -> mlir::Value {
        for (size_t i = 0; i < keywordCount; ++i) {
            if (llvm::StringRef(keywords[i]) == key) {
                return resolved_args[positionalCount + i];
            }
        }
        return {};
    };
    auto optionalValue = [&](size_t positionalIndex,
                             llvm::StringRef key) -> mlir::Value {
        if (positionalCount > positionalIndex) {
            return resolved_args[positionalIndex];
        }
        return keywordValue(key);
    };
    auto offsetValue = [&](size_t positionalIndex,
                           llvm::StringRef key) -> mlir::Value {
        if (auto value = optionalValue(positionalIndex, key)) {
            return value;
        }
        return createDefaultIndex(builder, location);
    };
    auto i32 = [&](mlir::Value value) {
        return castIntegerTo(builder, location, value, builder.getI32Type());
    };
    auto i16 = [&](mlir::Value value) {
        return castIntegerTo(builder, location, value, builder.getI16Type());
    };
    auto i64 = [&](mlir::Value value) {
        return castIntegerTo(builder, location, value, builder.getI64Type());
    };
    auto i1 = [&](mlir::Value value) {
        return castIntegerTo(builder, location, value, builder.getI1Type());
    };
    auto coordinates = [&](mlir::Value tuple) {
        llvm::SmallVector<mlir::Value> values;
        extractTupleValues(tuple, values);
        for (auto &value : values) {
            value = i32(value);
        }
        return values;
    };
    auto im2colOffsets = [&](mlir::Value tuple) {
        llvm::SmallVector<mlir::Value> values;
        if (tuple) {
            extractTupleValues(tuple, values);
        }
        for (auto &value : values) {
            value = i16(value);
        }
        return values;
    };

    switch (kind) {
    case CpAsyncBulkIntrinsicKind::CommitGroup:
        mlir::NVVM::CpAsyncBulkCommitGroupOp::create(builder, location);
        break;
    case CpAsyncBulkIntrinsicKind::WaitGroup: {
        auto group =
            static_cast<uint32_t>(*getConstantIntValue(resolved_args[0]));
        bool read = false;
        if (auto readValue = optionalValue(1, "read")) {
            read = *getConstantIntValue(readValue) != 0;
        }
        mlir::NVVM::CpAsyncBulkWaitGroupOp::create(builder, location, group,
                                                   read ? builder.getUnitAttr()
                                                        : mlir::UnitAttr{});
        break;
    }
    case CpAsyncBulkIntrinsicKind::Prefetch: {
        auto src = createPointerFromMemRef(builder, location, resolved_args[0],
                                           offsetValue(2, "src_offset_bytes"),
                                           mlir::NVVM::NVVMMemorySpace::Global);
        mlir::NVVM::CpAsyncBulkPrefetchOp::create(
            builder, location, src, i32(resolved_args[1]),
            i64(keywordValue("l2_cache_hint")));
        break;
    }
    case CpAsyncBulkIntrinsicKind::GlobalSharedCTA: {
        auto dst = createPointerFromMemRef(builder, location, resolved_args[0],
                                           offsetValue(3, "dst_offset_bytes"),
                                           mlir::NVVM::NVVMMemorySpace::Global);
        auto src = createPointerFromMemRef(builder, location, resolved_args[1],
                                           offsetValue(4, "src_offset_bytes"),
                                           mlir::NVVM::NVVMMemorySpace::Shared);
        mlir::NVVM::CpAsyncBulkSharedCTAToGlobalOp::create(
            builder, location, dst, src, i32(resolved_args[2]),
            i64(keywordValue("l2_cache_hint")), i16(keywordValue("byte_mask")));
        break;
    }
    case CpAsyncBulkIntrinsicKind::SharedClusterGlobal: {
        auto dst =
            createPointerFromMemRef(builder, location, resolved_args[0],
                                    offsetValue(4, "dst_offset_bytes"),
                                    mlir::NVVM::NVVMMemorySpace::SharedCluster);
        auto src = createPointerFromMemRef(builder, location, resolved_args[1],
                                           offsetValue(5, "src_offset_bytes"),
                                           mlir::NVVM::NVVMMemorySpace::Global);
        auto mbar =
            createSharedBarrierPointer(builder, location, resolved_args[2],
                                       offsetValue(6, "mbar_offset_bytes"));
        mlir::NVVM::CpAsyncBulkGlobalToSharedClusterOp::create(
            builder, location, dst, src, mbar, i32(resolved_args[3]),
            i16(keywordValue("multicast_mask")),
            i64(keywordValue("l2_cache_hint")));
        break;
    }
    case CpAsyncBulkIntrinsicKind::SharedClusterSharedCTA: {
        auto dst =
            createPointerFromMemRef(builder, location, resolved_args[0],
                                    offsetValue(4, "dst_offset_bytes"),
                                    mlir::NVVM::NVVMMemorySpace::SharedCluster);
        auto src = createPointerFromMemRef(builder, location, resolved_args[1],
                                           offsetValue(5, "src_offset_bytes"),
                                           mlir::NVVM::NVVMMemorySpace::Shared);
        auto mbar =
            createSharedBarrierPointer(builder, location, resolved_args[2],
                                       offsetValue(6, "mbar_offset_bytes"));
        mlir::NVVM::CpAsyncBulkSharedCTAToSharedClusterOp::create(
            builder, location, dst, src, mbar, i32(resolved_args[3]));
        break;
    }
    case CpAsyncBulkIntrinsicKind::TensorGlobalSharedCTA: {
        auto desc =
            createDescriptorPointer(builder, location, resolved_args[0]);
        auto src = createPointerFromMemRef(builder, location, resolved_args[1],
                                           offsetValue(3, "src_offset_bytes"),
                                           mlir::NVVM::NVVMMemorySpace::Shared);
        auto coords = coordinates(resolved_args[2]);
        mlir::Value predicate;
        if (auto value = keywordValue("predicate")) {
            predicate = i1(value);
        }
        mlir::NVVM::CpAsyncBulkTensorSharedCTAToGlobalOp::create(
            builder, location, desc, src, coords,
            i64(keywordValue("l2_cache_hint")), mlir::NVVM::TMAStoreMode::TILE,
            predicate);
        break;
    }
    case CpAsyncBulkIntrinsicKind::TensorPrefetch: {
        auto desc =
            createDescriptorPointer(builder, location, resolved_args[0]);
        auto coords = coordinates(resolved_args[1]);
        auto im2col = im2colOffsets(keywordValue("im2col_offsets"));
        auto mode = im2col.empty() ? mlir::NVVM::TMALoadMode::TILE
                                   : mlir::NVVM::TMALoadMode::IM2COL;
        mlir::NVVM::CpAsyncBulkTensorPrefetchOp::create(
            builder, location, desc, coords, im2col, mode,
            i64(keywordValue("l2_cache_hint")));
        break;
    }
    case CpAsyncBulkIntrinsicKind::TensorReduce: {
        auto desc =
            createDescriptorPointer(builder, location, resolved_args[0]);
        auto src = createPointerFromMemRef(builder, location, resolved_args[1],
                                           offsetValue(4, "src_offset_bytes"),
                                           mlir::NVVM::NVVMMemorySpace::Shared);
        auto coords = coordinates(resolved_args[2]);
        auto redKindValue =
            static_cast<uint32_t>(*getConstantIntValue(resolved_args[3]));
        auto redKind = *mlir::NVVM::symbolizeTMAReduxKind(redKindValue);
        mlir::NVVM::CpAsyncBulkTensorReduceOp::create(
            builder, location, desc, src, redKind,
            mlir::NVVM::TMAStoreMode::TILE, coords,
            i64(keywordValue("l2_cache_hint")));
        break;
    }
    case CpAsyncBulkIntrinsicKind::TensorSharedClusterGlobal: {
        auto dst =
            createPointerFromMemRef(builder, location, resolved_args[0],
                                    offsetValue(4, "dst_offset_bytes"),
                                    mlir::NVVM::NVVMMemorySpace::SharedCluster);
        auto desc =
            createDescriptorPointer(builder, location, resolved_args[1]);
        auto coords = coordinates(resolved_args[2]);
        auto mbar =
            createSharedBarrierPointer(builder, location, resolved_args[3],
                                       offsetValue(5, "mbar_offset_bytes"));
        auto im2col = im2colOffsets(keywordValue("im2col_offsets"));
        auto mode = im2col.empty() ? mlir::NVVM::TMALoadMode::TILE
                                   : mlir::NVVM::TMALoadMode::IM2COL;
        mlir::Value predicate;
        if (auto value = keywordValue("predicate")) {
            predicate = i1(value);
        }
        mlir::NVVM::CpAsyncBulkTensorGlobalToSharedClusterOp::create(
            builder, location, dst, desc, coords, mbar, im2col,
            i16(keywordValue("multicast_mask")),
            i64(keywordValue("l2_cache_hint")), mode,
            /*isCTAOnly=*/false, mlir::NVVM::CTAGroupKindAttr{}, predicate);
        break;
    }
    }

    return ctx->GetCurrentFunctionGenerator()
        ->GetExprGenerator()
        ->CreateVoidValue();
}

bool NVVMIntrinsic::CheckCpAsyncBulkFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args,
    CpAsyncBulkIntrinsicKind kind) const {
    const auto &keywords = call_expr->GetKeywords();
    if (resolved_args.size() < keywords.size()) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "cp_async_bulk internal argument mismatch";
        return false;
    }

    size_t keywordCount = keywords.size();
    size_t positionalCount = resolved_args.size() - keywordCount;

    auto fail = [&](llvm::StringRef message) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << message;
        return false;
    };
    auto checkPositionalCount = [&](size_t min, size_t max,
                                    llvm::StringRef message) {
        if (positionalCount < min || positionalCount > max) {
            return fail(message);
        }
        return true;
    };
    auto keywordAllowed = [&](llvm::StringRef keyword,
                              std::initializer_list<llvm::StringRef> allowed) {
        for (auto candidate : allowed) {
            if (keyword == candidate) {
                return true;
            }
        }
        return false;
    };
    auto checkKeywords =
        [&](std::initializer_list<llvm::StringRef> allowed) -> bool {
        for (auto name : keywords) {
            if (!keywordAllowed(name, allowed)) {
                std::string message =
                    "unsupported cp_async_bulk keyword argument '";
                message += llvm::StringRef(name).str();
                message += "'";
                return fail(message);
            }
        }
        return true;
    };
    auto checkPresent = [&](size_t index, llvm::StringRef name) {
        if (index >= resolved_args.size() || !resolved_args[index]) {
            std::string message = "failed to generate cp_async_bulk operand '";
            message += name.str();
            message += "'";
            return fail(message);
        }
        return true;
    };
    auto checkInt = [&](mlir::Value value, llvm::StringRef name) {
        if (value && !value.getType().isIntOrIndex()) {
            std::string message = "cp_async_bulk operand '";
            message += name.str();
            message += "' must be an integer/index";
            return fail(message);
        }
        return true;
    };
    auto checkMemRef = [&](mlir::Value value, llvm::StringRef name) {
        if (!value ||
            (!isMemRefLike(value.getType()) &&
             !mlir::isa<mlir::LLVM::LLVMPointerType>(value.getType()))) {
            std::string message = "cp_async_bulk operand '";
            message += name.str();
            message += "' must be a memref-like value";
            return fail(message);
        }
        return true;
    };
    auto checkDescriptor = [&](mlir::Value value, llvm::StringRef name) {
        if (!value ||
            (!mlir::isa<mlir::nvgpu::TensorMapDescriptorType>(
                 value.getType()) &&
             !mlir::isa<mlir::LLVM::LLVMPointerType>(value.getType()))) {
            std::string message = "cp_async_bulk operand '";
            message += name.str();
            message += "' must be a TMA descriptor";
            return fail(message);
        }
        return true;
    };
    auto checkBarrier = [&](mlir::Value value, llvm::StringRef name) {
        if (!value ||
            (!mlir::isa<mlir::nvgpu::MBarrierGroupType>(value.getType()) &&
             !isMemRefLike(value.getType()) &&
             !mlir::isa<mlir::LLVM::LLVMPointerType>(value.getType()))) {
            std::string message = "cp_async_bulk operand '";
            message += name.str();
            message += "' must be an mbarrier or shared pointer";
            return fail(message);
        }
        return true;
    };
    auto checkTuple = [&](mlir::Value value, llvm::StringRef name) {
        llvm::SmallVector<mlir::Value> values;
        if (!value || !extractTupleValues(value, values) || values.empty()) {
            std::string message = "cp_async_bulk operand '";
            message += name.str();
            message += "' must be a non-empty coordinate tuple";
            return fail(message);
        }
        for (auto elem : values) {
            if (!checkInt(elem, name)) {
                return false;
            }
        }
        return true;
    };
    auto checkOptionalKeywordInts =
        [&](std::initializer_list<llvm::StringRef> intNames) -> bool {
        for (size_t i = 0; i < keywordCount; ++i) {
            llvm::StringRef name = keywords[i];
            for (auto intName : intNames) {
                if (name == intName &&
                    !checkInt(resolved_args[positionalCount + i], name)) {
                    return false;
                }
            }
        }
        return true;
    };

    switch (kind) {
    case CpAsyncBulkIntrinsicKind::CommitGroup:
        return checkPositionalCount(
                   0, 0, "cp_async_bulk_commit_group takes no arguments") &&
               checkKeywords({});
    case CpAsyncBulkIntrinsicKind::WaitGroup: {
        if (!checkPositionalCount(
                1, 2,
                "cp_async_bulk_wait_group requires group and optional read") ||
            !checkKeywords({"read"}) || !checkPresent(0, "group") ||
            !checkInt(resolved_args[0], "group")) {
            return false;
        }
        auto group = getConstantIntValue(resolved_args[0]);
        if (!group || *group < 0) {
            return fail("cp_async_bulk_wait_group group must be a non-negative "
                        "compile-time constant");
        }
        if (positionalCount > 1 && (!checkInt(resolved_args[1], "read") ||
                                    !getConstantIntValue(resolved_args[1]))) {
            return fail("cp_async_bulk_wait_group read must be a compile-time "
                        "constant");
        }
        return checkOptionalKeywordInts({"read"});
    }
    case CpAsyncBulkIntrinsicKind::Prefetch:
        if (!checkPositionalCount(2, 3,
                                  "cp_async_bulk_prefetch requires src, size "
                                  "and optional src_offset_bytes") ||
            !checkKeywords({"src_offset_bytes", "l2_cache_hint"}) ||
            !checkMemRef(resolved_args[0], "src") ||
            !checkInt(resolved_args[1], "size")) {
            return false;
        }
        if (positionalCount > 2 &&
            !checkInt(resolved_args[2], "src_offset_bytes")) {
            return false;
        }
        return checkOptionalKeywordInts({"src_offset_bytes", "l2_cache_hint"});
    case CpAsyncBulkIntrinsicKind::GlobalSharedCTA:
        if (!checkPositionalCount(3, 5,
                                  "cp_async_bulk_global_shared_cta requires "
                                  "dst, src, size and optional offsets") ||
            !checkKeywords({"dst_offset_bytes", "src_offset_bytes",
                            "l2_cache_hint", "byte_mask"}) ||
            !checkMemRef(resolved_args[0], "dst") ||
            !checkMemRef(resolved_args[1], "src") ||
            !checkInt(resolved_args[2], "size")) {
            return false;
        }
        for (size_t i = 3; i < positionalCount; ++i) {
            if (!checkInt(resolved_args[i],
                          i == 3 ? "dst_offset_bytes" : "src_offset_bytes")) {
                return false;
            }
        }
        return checkOptionalKeywordInts({"dst_offset_bytes", "src_offset_bytes",
                                         "l2_cache_hint", "byte_mask"});
    case CpAsyncBulkIntrinsicKind::SharedClusterGlobal:
    case CpAsyncBulkIntrinsicKind::SharedClusterSharedCTA: {
        bool isClusterGlobal =
            kind == CpAsyncBulkIntrinsicKind::SharedClusterGlobal;
        if (!checkPositionalCount(
                4, 7,
                isClusterGlobal
                    ? "cp_async_bulk_shared_cluster_global requires dst, src, "
                      "mbar, size and optional offsets"
                    : "cp_async_bulk_shared_cluster_shared_cta requires dst, "
                      "src, mbar, size and optional offsets") ||
            !checkKeywords(isClusterGlobal
                               ? std::initializer_list<
                                     llvm::StringRef>{"dst_offset_bytes",
                                                      "src_offset_bytes",
                                                      "mbar_offset_bytes",
                                                      "multicast_mask",
                                                      "l2_cache_hint"}
                               : std::initializer_list<
                                     llvm::StringRef>{"dst_offset_bytes",
                                                      "src_offset_bytes",
                                                      "mbar_offset_bytes"}) ||
            !checkMemRef(resolved_args[0], "dst") ||
            !checkMemRef(resolved_args[1], "src") ||
            !checkBarrier(resolved_args[2], "mbar") ||
            !checkInt(resolved_args[3], "size")) {
            return false;
        }
        for (size_t i = 4; i < positionalCount; ++i) {
            if (!checkInt(resolved_args[i], "offset")) {
                return false;
            }
        }
        return checkOptionalKeywordInts({"dst_offset_bytes", "src_offset_bytes",
                                         "mbar_offset_bytes", "multicast_mask",
                                         "l2_cache_hint"});
    }
    case CpAsyncBulkIntrinsicKind::TensorGlobalSharedCTA:
        if (!checkPositionalCount(
                3, 4,
                "cp_async_bulk_tensor_global_shared_cta requires desc, src, "
                "coords and optional src_offset_bytes") ||
            !checkKeywords(
                {"src_offset_bytes", "l2_cache_hint", "predicate"}) ||
            !checkDescriptor(resolved_args[0], "desc") ||
            !checkMemRef(resolved_args[1], "src") ||
            !checkTuple(resolved_args[2], "coords")) {
            return false;
        }
        if (positionalCount > 3 &&
            !checkInt(resolved_args[3], "src_offset_bytes")) {
            return false;
        }
        return checkOptionalKeywordInts(
            {"src_offset_bytes", "l2_cache_hint", "predicate"});
    case CpAsyncBulkIntrinsicKind::TensorPrefetch:
        if (!checkPositionalCount(
                2, 2,
                "cp_async_bulk_tensor_prefetch requires desc and coords") ||
            !checkKeywords({"im2col_offsets", "l2_cache_hint"}) ||
            !checkDescriptor(resolved_args[0], "desc") ||
            !checkTuple(resolved_args[1], "coords")) {
            return false;
        }
        for (size_t i = 0; i < keywordCount; ++i) {
            if (llvm::StringRef(keywords[i]) == "im2col_offsets" &&
                resolved_args[positionalCount + i] &&
                !checkTuple(resolved_args[positionalCount + i],
                            "im2col_offsets")) {
                return false;
            }
        }
        return checkOptionalKeywordInts({"l2_cache_hint"});
    case CpAsyncBulkIntrinsicKind::TensorReduce: {
        if (!checkPositionalCount(
                4, 5,
                "cp_async_bulk_tensor_reduce requires desc, src, coords, "
                "red_kind and optional src_offset_bytes") ||
            !checkKeywords({"src_offset_bytes", "l2_cache_hint"}) ||
            !checkDescriptor(resolved_args[0], "desc") ||
            !checkMemRef(resolved_args[1], "src") ||
            !checkTuple(resolved_args[2], "coords") ||
            !checkInt(resolved_args[3], "red_kind")) {
            return false;
        }
        auto redKind = getConstantIntValue(resolved_args[3]);
        if (!redKind || *redKind < 0 ||
            !mlir::NVVM::symbolizeTMAReduxKind(
                static_cast<uint32_t>(*redKind))) {
            return fail(
                "cp_async_bulk_tensor_reduce red_kind must be one of 0..7");
        }
        if (positionalCount > 4 &&
            !checkInt(resolved_args[4], "src_offset_bytes")) {
            return false;
        }
        return checkOptionalKeywordInts({"src_offset_bytes", "l2_cache_hint"});
    }
    case CpAsyncBulkIntrinsicKind::TensorSharedClusterGlobal:
        if (!checkPositionalCount(
                4, 6,
                "cp_async_bulk_tensor_shared_cluster_global requires dst, "
                "desc, coords, mbar and optional offsets") ||
            !checkKeywords({"dst_offset_bytes", "mbar_offset_bytes",
                            "im2col_offsets", "multicast_mask", "l2_cache_hint",
                            "predicate"}) ||
            !checkMemRef(resolved_args[0], "dst") ||
            !checkDescriptor(resolved_args[1], "desc") ||
            !checkTuple(resolved_args[2], "coords") ||
            !checkBarrier(resolved_args[3], "mbar")) {
            return false;
        }
        for (size_t i = 4; i < positionalCount; ++i) {
            if (!checkInt(resolved_args[i], "offset")) {
                return false;
            }
        }
        for (size_t i = 0; i < keywordCount; ++i) {
            if (llvm::StringRef(keywords[i]) == "im2col_offsets" &&
                resolved_args[positionalCount + i] &&
                !checkTuple(resolved_args[positionalCount + i],
                            "im2col_offsets")) {
                return false;
            }
        }
        return checkOptionalKeywordInts({"dst_offset_bytes",
                                         "mbar_offset_bytes", "multicast_mask",
                                         "l2_cache_hint", "predicate"});
    }

    llvm_unreachable("unknown cp.async.bulk intrinsic kind");
}

mlir::Value NVVMIntrinsic::CreateMakeTMADescriptorFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    auto &builder = ctx->GetCurrentFunctionGenerator()->GetBuilder();
    auto location = builder.getUnknownLoc();

    if (!CheckMakeTMADescriptorFunction(call_expr, ctx, resolved_args)) {
        return nullptr;
    }

    auto tensorType =
        mlir::dyn_cast<cf::MemRefType>(resolved_args[0].getType());
    if (!tensorType) {
        return nullptr;
    }

    auto layoutOp = resolved_args[1].getDefiningOp<cf::MakeLayoutOp>();
    llvm::SmallVector<int64_t> smemDims;
    llvm::SmallVector<int64_t> smemStrides;
    if (!layoutOp ||
        !extractConstantTupleValues(layoutOp.getDims(), smemDims) ||
        !extractConstantTupleValues(layoutOp.getStride(), smemStrides) ||
        smemDims.size() != smemStrides.size()) {
        return nullptr;
    }

    auto workgroupMemorySpace = mlir::IntegerAttr::get(
        mlir::IntegerType::get(builder.getContext(), 64), 3);
    auto descriptorTensorType = mlir::MemRefType::get(
        smemDims, tensorType.getElementType(),
        mlir::StridedLayoutAttr::get(builder.getContext(), 0, smemStrides),
        workgroupMemorySpace);

    auto resultType = mlir::nvgpu::TensorMapDescriptorType::get(
        builder.getContext(), descriptorTensorType,
        mlir::nvgpu::TensorMapSwizzleKind::SWIZZLE_NONE,
        mlir::nvgpu::TensorMapL2PromoKind::L2PROMO_NONE,
        mlir::nvgpu::TensorMapOOBKind::OOB_ZERO,
        mlir::nvgpu::TensorMapInterleaveKind::INTERLEAVE_NONE);

    auto descriptor = cf::NVVMTMADescriptorOp::create(
        builder, location, resultType, resolved_args[0], resolved_args[1]);
    return descriptor.getResult();
}

bool NVVMIntrinsic::CheckMakeTMADescriptorFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    if (resolved_args.size() != 2) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "make_tma_descriptor requires exactly 2 arguments: tensor, "
               "smem_layout";
        return false;
    }

    if (!resolved_args[0]) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "Failed to generate tensor operand for make_tma_descriptor";
        return false;
    }

    if (!resolved_args[1]) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "Failed to generate smem_layout operand for make_tma_descriptor";
        return false;
    }

    auto tensorType =
        mlir::dyn_cast<cf::MemRefType>(resolved_args[0].getType());
    if (!tensorType) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "make_tma_descriptor expects a memref tensor operand";
        return false;
    }

    auto layoutOp = resolved_args[1].getDefiningOp<cf::MakeLayoutOp>();
    if (!layoutOp) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "make_tma_descriptor expects smem_layout from make_layout()";
        return false;
    }

    auto dimsTuple = layoutOp.getDims().getDefiningOp<cf::MakeIntTupleOp>();
    if (!dimsTuple || dimsTuple.getNumElements() == 0) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "make_tma_descriptor requires a non-empty smem_layout";
        return false;
    }

    llvm::SmallVector<int64_t> smemDims;
    llvm::SmallVector<int64_t> smemStrides;
    if (!extractConstantTupleValues(layoutOp.getDims(), smemDims) ||
        !extractConstantTupleValues(layoutOp.getStride(), smemStrides) ||
        smemDims.size() != smemStrides.size()) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "make_tma_descriptor requires a static smem_layout";
        return false;
    }

    return true;
}

mlir::Value NVVMIntrinsic::CreateTMAFenceFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    auto &builder = ctx->GetCurrentFunctionGenerator()->GetBuilder();
    auto location = builder.getUnknownLoc();

    if (!CheckTMAFenceFunction(call_expr, ctx, resolved_args)) {
        return nullptr;
    }

    cf::NVVMTMAFenceOp::create(builder, location,
                               mlir::ValueRange{resolved_args[0]},
                               mlir::ArrayRef<mlir::NamedAttribute>{});
    return ctx->GetCurrentFunctionGenerator()
        ->GetExprGenerator()
        ->CreateVoidValue();
}

bool NVVMIntrinsic::CheckTMAFenceFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    if (resolved_args.size() != 1) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "tma_fence requires exactly 1 argument: desc";
        return false;
    }

    if (!resolved_args[0]) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "Failed to generate desc operand for tma_fence";
        return false;
    }

    if (!mlir::isa<mlir::nvgpu::TensorMapDescriptorType>(
            resolved_args[0].getType())) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "tma_fence desc operand must be a TMA descriptor";
        return false;
    }

    return true;
}

mlir::Value NVVMIntrinsic::CreateTMALoadFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    auto &builder = ctx->GetCurrentFunctionGenerator()->GetBuilder();
    auto location = builder.getUnknownLoc();

    if (!CheckTMALoadFunction(call_expr, ctx, resolved_args)) {
        return nullptr;
    }

    const auto &keywords = call_expr->GetKeywords();
    size_t keywordCount = keywords.size();
    size_t positionalCount = resolved_args.size() - keywordCount;

    mlir::Value mbarId =
        positionalCount > 4
            ? resolved_args[4]
            : mlir::arith::ConstantIndexOp::create(builder, location, 0)
                  .getResult();
    mlir::Value predicate =
        positionalCount > 5
            ? resolved_args[5]
            : mlir::arith::ConstantIntOp::create(builder, location, 1, 1)
                  .getResult();
    mlir::Value multicastMask =
        positionalCount > 6
            ? resolved_args[6]
            : mlir::arith::ConstantIntOp::create(builder, location, -1, 32)
                  .getResult();

    for (size_t i = 0; i < keywordCount; ++i) {
        mlir::Value value = resolved_args[positionalCount + i];
        llvm::StringRef name = keywords[i];
        if (name == "mbar_id") {
            mbarId = value;
        } else if (name == "predicate") {
            predicate = value;
        } else if (name == "multicast_mask") {
            if (value) {
                multicastMask = value;
            }
        }
    }

    cf::NVVMTMALoadOp::create(
        builder, location,
        mlir::ValueRange{resolved_args[0], resolved_args[1], resolved_args[2],
                         resolved_args[3], mbarId, predicate, multicastMask},
        mlir::ArrayRef<mlir::NamedAttribute>{});
    return ctx->GetCurrentFunctionGenerator()
        ->GetExprGenerator()
        ->CreateVoidValue();
}

bool NVVMIntrinsic::CheckTMALoadFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    const auto &keywords = call_expr->GetKeywords();
    if (resolved_args.size() < keywords.size()) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "tma_load internal argument mismatch";
        return false;
    }

    size_t keywordCount = keywords.size();
    size_t positionalCount = resolved_args.size() - keywordCount;
    if (positionalCount < 4 || positionalCount > 7) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "tma_load requires dst, desc, coords, barrier and optional "
               "mbar_id, predicate, multicast_mask";
        return false;
    }

    for (auto name : keywords) {
        if (name != "mbar_id" && name != "predicate" &&
            name != "multicast_mask") {
            ctx->diagnostic_manager->Report(
                basic::DiagnosticCode::kUnimplemented,
                call_expr->GetSourceRange().getBegin())
                << "tma_load got unsupported keyword argument '" << name << "'";
            return false;
        }
    }

    auto requireArg = [&](size_t index, llvm::StringRef name) -> bool {
        if (index >= resolved_args.size() || !resolved_args[index]) {
            ctx->diagnostic_manager->Report(
                basic::DiagnosticCode::kUnimplemented,
                call_expr->GetSourceRange().getBegin())
                << "Failed to generate " << name << " operand for tma_load";
            return false;
        }
        return true;
    };

    if (!requireArg(0, "dst") || !requireArg(1, "desc") ||
        !requireArg(2, "coords") || !requireArg(3, "barrier")) {
        return false;
    }

    if (!mlir::isa<cf::MemRefType>(resolved_args[0].getType())) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "tma_load dst operand must be a memref";
        return false;
    }

    if (!mlir::isa<mlir::nvgpu::TensorMapDescriptorType>(
            resolved_args[1].getType())) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "tma_load desc operand must be a TMA descriptor";
        return false;
    }

    if (!mlir::isa<mlir::nvgpu::MBarrierGroupType>(
            resolved_args[3].getType())) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "tma_load barrier operand must be of type mbarrier_group_t";
        return false;
    }

    auto checkIntArg = [&](mlir::Value value, llvm::StringRef name) -> bool {
        if (value && !value.getType().isIntOrIndex()) {
            ctx->diagnostic_manager->Report(
                basic::DiagnosticCode::kUnimplemented,
                call_expr->GetSourceRange().getBegin())
                << "tma_load " << name << " operand must be an integer type";
            return false;
        }
        return true;
    };

    for (size_t i = 4; i < positionalCount; ++i) {
        if (!checkIntArg(resolved_args[i], i == 4   ? "mbar_id"
                                           : i == 5 ? "predicate"
                                                    : "multicast_mask")) {
            return false;
        }
    }
    for (size_t i = 0; i < keywordCount; ++i) {
        auto name = llvm::StringRef(keywords[i]);
        auto value = resolved_args[positionalCount + i];
        if (name == "mbar_id" || name == "predicate" ||
            (name == "multicast_mask" && value)) {
            if (!checkIntArg(value, name)) {
                return false;
            }
        }
    }

    return true;
}

mlir::Value NVVMIntrinsic::CreateTMAStoreFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    auto &builder = ctx->GetCurrentFunctionGenerator()->GetBuilder();
    auto location = builder.getUnknownLoc();

    if (!CheckTMAStoreFunction(call_expr, ctx, resolved_args)) {
        return nullptr;
    }

    const auto &keywords = call_expr->GetKeywords();
    size_t keywordCount = keywords.size();
    size_t positionalCount = resolved_args.size() - keywordCount;

    mlir::Value predicate =
        positionalCount > 3
            ? resolved_args[3]
            : mlir::arith::ConstantIntOp::create(builder, location, 1, 1)
                  .getResult();

    for (size_t i = 0; i < keywordCount; ++i) {
        mlir::Value value = resolved_args[positionalCount + i];
        llvm::StringRef name = keywords[i];
        if (name == "predicate") {
            predicate = value;
        }
    }

    cf::NVVMTMAStoreOp::create(builder, location,
                               mlir::ValueRange{resolved_args[0],
                                                resolved_args[1],
                                                resolved_args[2], predicate},
                               mlir::ArrayRef<mlir::NamedAttribute>{});
    return ctx->GetCurrentFunctionGenerator()
        ->GetExprGenerator()
        ->CreateVoidValue();
}

bool NVVMIntrinsic::CheckTMAStoreFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    const auto &keywords = call_expr->GetKeywords();
    if (resolved_args.size() < keywords.size()) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "tma_store internal argument mismatch";
        return false;
    }

    size_t keywordCount = keywords.size();
    size_t positionalCount = resolved_args.size() - keywordCount;
    if (positionalCount < 3 || positionalCount > 4) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "tma_store requires src, desc, coords and optional predicate";
        return false;
    }

    for (auto name : keywords) {
        if (name != "predicate") {
            ctx->diagnostic_manager->Report(
                basic::DiagnosticCode::kUnimplemented,
                call_expr->GetSourceRange().getBegin())
                << "tma_store got unsupported keyword argument '" << name
                << "'";
            return false;
        }
    }

    auto requireArg = [&](size_t index, llvm::StringRef name) -> bool {
        if (index >= resolved_args.size() || !resolved_args[index]) {
            ctx->diagnostic_manager->Report(
                basic::DiagnosticCode::kUnimplemented,
                call_expr->GetSourceRange().getBegin())
                << "Failed to generate " << name << " operand for tma_store";
            return false;
        }
        return true;
    };

    if (!requireArg(0, "src") || !requireArg(1, "desc") ||
        !requireArg(2, "coords")) {
        return false;
    }

    if (!mlir::isa<cf::MemRefType>(resolved_args[0].getType())) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "tma_store src operand must be a memref";
        return false;
    }

    if (!mlir::isa<mlir::nvgpu::TensorMapDescriptorType>(
            resolved_args[1].getType())) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "tma_store desc operand must be a TMA descriptor";
        return false;
    }

    auto checkIntArg = [&](mlir::Value value, llvm::StringRef name) -> bool {
        if (value && !value.getType().isIntOrIndex()) {
            ctx->diagnostic_manager->Report(
                basic::DiagnosticCode::kUnimplemented,
                call_expr->GetSourceRange().getBegin())
                << "tma_store " << name << " operand must be an integer type";
            return false;
        }
        return true;
    };

    if (positionalCount > 3 && !checkIntArg(resolved_args[3], "predicate")) {
        return false;
    }
    for (size_t i = 0; i < keywordCount; ++i) {
        if (!checkIntArg(resolved_args[positionalCount + i], keywords[i])) {
            return false;
        }
    }

    return true;
}

// Factory function to create NVVM intrinsic module
std::unique_ptr<NamedModule> CreateNVVMIntrinsicModule() {
    return std::make_unique<NVVMIntrinsic>();
}

} // namespace causalflow::avelang::ir
