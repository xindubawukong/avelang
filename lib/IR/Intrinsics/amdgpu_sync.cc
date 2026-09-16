#include "amdgpu_sync.h"
#include "IR/constant_folder.h"
#include "IR/generator_context.h"
#include "IR/mlir_generator_impl.h"
#include "IR/type_system.h"
#include <llvm/ADT/STLExtras.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>

namespace causalflow::avelang::ir::intrinsics {
namespace {
using Args = llvm::ArrayRef<mlir::Value>;
bool Error(ast::Call *call, GeneratorContext *ctx, llvm::StringRef message) {
    ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                    call->GetSourceRange().getBegin())
        << message;
    return false;
}
bool IsBufferInt(mlir::Value value) {
    return value.getType().isIndex() || value.getType().isInteger(32) ||
           value.getType().isInteger(64);
}
bool Count(Args args, unsigned n) {
    return args.size() == n &&
           llvm::all_of(args, [](auto v) { return bool(v); });
}
mlir::Value I32(mlir::OpBuilder &b, mlir::Location loc, mlir::Value value) {
    if (value.getType().isIndex())
        return mlir::arith::IndexCastOp::create(b, loc, b.getI32Type(), value);
    auto width = mlir::cast<mlir::IntegerType>(value.getType()).getWidth();
    if (width > 32)
        return mlir::arith::TruncIOp::create(b, loc, b.getI32Type(), value);
    if (width < 32)
        return mlir::arith::ExtUIOp::create(b, loc, b.getI32Type(), value);
    return value;
}
auto Void(GeneratorContext *ctx) {
    return ctx->GetCurrentFunctionGenerator()
        ->GetExprGenerator()
        ->CreateVoidValue();
}
} // namespace
SymbolScope::Function BufferAtomicI32(bool bitwiseOr) {
    return {
        [bitwiseOr](ast::Call *call, GeneratorContext *ctx, Args args) {
            auto &b = ctx->GetCurrentFunctionGenerator()->GetBuilder();
            auto loc = ctx->GetMLIRLocation(b.getContext(), call);
            auto aux = mlir::arith::ConstantIntOp::create(
                b, loc, *ConstantFolder::FoldIntValue(args[4]), 32);
            auto op = mlir::LLVM::CallIntrinsicOp::create(
                b, loc, b.getI32Type(),
                b.getStringAttr(bitwiseOr
                                    ? "llvm.amdgcn.raw.buffer.atomic.or"
                                    : "llvm.amdgcn.raw.buffer.atomic.add"),
                mlir::ValueRange{I32(b, loc, args[0]), args[1],
                                 I32(b, loc, args[2]), I32(b, loc, args[3]),
                                 aux});
            SetTypeInfo(op.getResult(0), TypeInfo{true});
            return op.getResult(0);
        },
        [](ast::Call *call, GeneratorContext *ctx, Args args) {
            if (!Count(args, 5))
                return Error(call, ctx,
                             "raw_buffer_atomic expects value, resource, "
                             "byte_offset, soffset, aux");
            auto r = mlir::dyn_cast<mlir::VectorType>(args[1].getType());
            auto aux = ConstantFolder::FoldIntValue(args[4]);
            if (!IsBufferInt(args[0]) || !r ||
                r.getShape() != llvm::ArrayRef<int64_t>{4} ||
                !r.getElementType().isInteger(32) || !IsBufferInt(args[2]) ||
                !IsBufferInt(args[3]) || !aux || (*aux != 0 && *aux != 16))
                return Error(
                    call, ctx,
                    "raw_buffer_atomic expects i32/i64/index value/offsets, "
                    "vector<4xi32> resource and aux 0 (agent) or 16 (system)");
            return true;
        }};
}
SymbolScope::Function CompilerBarrier() {
    return {
        [](ast::Call *call, GeneratorContext *ctx, Args) {
            auto &b = ctx->GetCurrentFunctionGenerator()->GetBuilder();
            mlir::LLVM::InlineAsmOp::create(
                b, ctx->GetMLIRLocation(b.getContext(), call), mlir::Type(),
                mlir::ValueRange{}, "", "~{memory}", true, false,
                mlir::LLVM::tailcallkind::TailCallKind::None, nullptr, nullptr);
            return Void(ctx);
        },
        [](ast::Call *call, GeneratorContext *ctx, Args args) {
            return Count(args, 0) ||
                   Error(call, ctx, "compiler_barrier expects no arguments");
        }};
}
SymbolScope::Function Sleep() {
    return {
        [](ast::Call *call, GeneratorContext *ctx, Args args) {
            auto &b = ctx->GetCurrentFunctionGenerator()->GetBuilder();
            auto cycles = *ConstantFolder::FoldIntValue(args[0]);
            mlir::LLVM::InlineAsmOp::create(
                b, ctx->GetMLIRLocation(b.getContext(), call), mlir::Type(),
                mlir::ValueRange{}, "s_sleep " + std::to_string(cycles),
                "~{memory}", true, false,
                mlir::LLVM::tailcallkind::TailCallKind::None, nullptr, nullptr);
            return Void(ctx);
        },
        [](ast::Call *call, GeneratorContext *ctx, Args args) {
            if (!Count(args, 1))
                return Error(call, ctx,
                             "s_sleep expects a constant in [0, 15]");
            auto v = ConstantFolder::FoldIntValue(args[0]);
            return (v && *v >= 0 && *v <= 15) ||
                   Error(call, ctx, "s_sleep expects a constant in [0, 15]");
        }};
}
SymbolScope::Function Fence() {
    return {[](ast::Call *call, GeneratorContext *ctx, Args args) {
                auto &b = ctx->GetCurrentFunctionGenerator()->GetBuilder();
                auto order = *ConstantFolder::FoldIntValue(args[0]);
                auto scope = *ConstantFolder::FoldIntValue(args[1]);
                const auto ordering =
                    order == 0   ? mlir::LLVM::AtomicOrdering::acquire
                    : order == 1 ? mlir::LLVM::AtomicOrdering::release
                                 : mlir::LLVM::AtomicOrdering::acq_rel;
                mlir::LLVM::FenceOp::create(
                    b, ctx->GetMLIRLocation(b.getContext(), call), ordering,
                    scope == 0   ? "workgroup"
                    : scope == 1 ? "agent"
                                 : "");
                return Void(ctx);
            },
            [](ast::Call *call, GeneratorContext *ctx, Args args) {
                if (!Count(args, 2))
                    return Error(call, ctx, "fence expects ordering and scope");
                auto order = ConstantFolder::FoldIntValue(args[0]);
                auto scope = ConstantFolder::FoldIntValue(args[1]);
                return (order && scope && *order >= 0 && *order <= 2 &&
                        *scope >= 0 && *scope <= 2) ||
                       Error(call, ctx,
                             "fence ordering: 0 acquire, 1 release, 2 "
                             "acquire-release; scope: 0 workgroup, 1 agent, 2 "
                             "system");
            }};
}
} // namespace causalflow::avelang::ir::intrinsics
