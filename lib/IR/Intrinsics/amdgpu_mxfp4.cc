#include "amdgpu_mxfp4.h"
#include "IR/constant_folder.h"
#include "IR/generator_context.h"
#include "IR/mlir_generator_impl.h"
#include "IR/type_system.h"

#include <llvm/ADT/STLExtras.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/IR/Builders.h>
#include <utility>

namespace causalflow::avelang::ir::intrinsics {
namespace {

using Args = llvm::ArrayRef<mlir::Value>;

bool Error(ast::Call *call, GeneratorContext *ctx, llvm::StringRef message) {
    ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                    call->GetSourceRange().getBegin())
        << message;
    return false;
}

bool IsI32(mlir::Value value) { return value.getType().isInteger(32); }

bool IsVector4(mlir::Value value, bool integer) {
    auto type = mlir::dyn_cast<mlir::VectorType>(value.getType());
    return type && type.getShape() == llvm::ArrayRef<int64_t>{4} &&
           (integer ? type.getElementType().isInteger(32)
                    : type.getElementType().isF32());
}

bool IsSelector(mlir::Value value) {
    auto selector = ConstantFolder::FoldIntValue(value);
    return value.getType().isIntOrIndex() &&
           (!selector || (*selector >= 0 && *selector < 4));
}

bool CheckArgs(ast::Call *call, Args args, unsigned count) {
    return call->GetArgs().size() == count && args.size() == count &&
           llvm::all_of(args, [](mlir::Value value) { return bool(value); });
}

mlir::Value Constant(mlir::OpBuilder &builder, mlir::Location loc,
                     int64_t value) {
    return mlir::arith::ConstantIntOp::create(builder, loc, value, 32);
}

mlir::Value Intrinsic(mlir::OpBuilder &builder, mlir::Location loc,
                      mlir::Type type, llvm::StringRef name, Args operands) {
    return mlir::LLVM::CallIntrinsicOp::create(
               builder, loc, type, builder.getStringAttr(name), operands)
        .getResult(0);
}

std::pair<mlir::Value, mlir::Value> SelectScale(mlir::OpBuilder &builder,
                                                mlir::Location loc,
                                                mlir::Value scale,
                                                mlir::Value selector) {
    if (auto value = ConstantFolder::FoldIntValue(selector))
        return {scale, Constant(builder, loc, *value)};
    // Loop induction variables need not be constant during frontend lowering.
    // Select the byte explicitly in that case; CSE can share it across MFMAs.
    if (selector.getType().isIndex())
        selector = mlir::arith::IndexCastOp::create(
            builder, loc, builder.getI32Type(), selector);
    else if (auto integer =
                 mlir::dyn_cast<mlir::IntegerType>(selector.getType())) {
        if (integer.getWidth() > 32)
            selector = mlir::arith::TruncIOp::create(
                builder, loc, builder.getI32Type(), selector);
        else if (integer.getWidth() < 32)
            selector = mlir::arith::ExtUIOp::create(
                builder, loc, builder.getI32Type(), selector);
    }
    auto lowBits = mlir::arith::AndIOp::create(builder, loc, selector,
                                               Constant(builder, loc, 3));
    auto shift = mlir::arith::ShLIOp::create(builder, loc, lowBits,
                                             Constant(builder, loc, 3));
    return {mlir::arith::ShRUIOp::create(builder, loc, scale, shift),
            Constant(builder, loc, 0)};
}

} // namespace

SymbolScope::Function MxFp4ScaledMfma() {
    return {
        [](ast::Call *call, GeneratorContext *ctx, Args args) {
            auto &builder = ctx->GetCurrentFunctionGenerator()->GetBuilder();
            auto loc = ctx->GetMLIRLocation(builder.getContext(), call);
            auto format = Constant(builder, loc, 4); // E2M1 FP4
            auto [scaleA, selA] = SelectScale(builder, loc, args[1], args[5]);
            auto [scaleB, selB] = SelectScale(builder, loc, args[3], args[6]);
            // The intrinsic overload accepts the four packed i32 registers
            // required by FP4 directly. Each selected scale byte is E8M0.
            return Intrinsic(builder, loc, args[4].getType(),
                             "llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4",
                             {args[0], args[2], args[4], format, format, selA,
                              scaleA, selB, scaleB});
        },
        [](ast::Call *call, GeneratorContext *ctx, Args args) {
            if (!CheckArgs(call, args, 7) || !IsVector4(args[0], true) ||
                !IsI32(args[1]) || !IsVector4(args[2], true) ||
                !IsI32(args[3]) || !IsVector4(args[4], false) ||
                !IsSelector(args[5]) || !IsSelector(args[6]))
                return Error(
                    call, ctx,
                    "mfma_scale_16x16x128_fp4(a, scale_a, b, scale_b, c, "
                    "opsel_a, opsel_b) expects vector<4xi32> operands, i32 "
                    "scales, vector<4xf32> accumulator and scale selectors "
                    "in [0, 3]");
            return true;
        }};
}

SymbolScope::Function MxFp4Pack() {
    return {[](ast::Call *call, GeneratorContext *ctx, Args args) {
                auto &builder =
                    ctx->GetCurrentFunctionGenerator()->GetBuilder();
                auto loc = ctx->GetMLIRLocation(builder.getContext(), call);
                auto selector = Constant(
                    builder, loc, *ConstantFolder::FoldIntValue(args[4]));
                auto result =
                    Intrinsic(builder, loc, builder.getI32Type(),
                              "llvm.amdgcn.cvt.scalef32.pk.fp4.f32",
                              {args[0], args[1], args[2], args[3], selector});
                SetTypeInfo(result, TypeInfo{true});
                return result;
            },
            [](ast::Call *call, GeneratorContext *ctx, Args args) {
                if (!CheckArgs(call, args, 5) || !IsI32(args[0]) ||
                    !args[1].getType().isF32() || !args[2].getType().isF32() ||
                    !args[3].getType().isF32() || !IsSelector(args[4]) ||
                    !ConstantFolder::FoldIntValue(args[4]))
                    return Error(
                        call, ctx,
                        "cvt_scalef32_pk_fp4_f32(old, a, b, scale, byte_sel) "
                        "expects i32 old, f32 values/scale and a constant "
                        "byte_sel in [0, 3]");
                return true;
            }};
}

SymbolScope::Function BufferAtomicAddBf16x2() {
    return {[](ast::Call *call, GeneratorContext *ctx, Args args) {
                auto &builder =
                    ctx->GetCurrentFunctionGenerator()->GetBuilder();
                auto loc = ctx->GetMLIRLocation(builder.getContext(), call);
                Intrinsic(builder, loc, args[0].getType(),
                          "llvm.amdgcn.raw.buffer.atomic.fadd",
                          {args[0], args[1], args[2], Constant(builder, loc, 0),
                           Constant(builder, loc, 0)});
                return ctx->GetCurrentFunctionGenerator()
                    ->GetExprGenerator()
                    ->CreateVoidValue();
            },
            [](ast::Call *call, GeneratorContext *ctx, Args args) {
                if (!CheckArgs(call, args, 3))
                    return Error(call, ctx,
                                 "raw_buffer_atomic_add_bf16x2 expects value, "
                                 "resource and byte offset");
                auto type = mlir::dyn_cast<mlir::VectorType>(args[0].getType());
                if (!type || type.getShape() != llvm::ArrayRef<int64_t>{2} ||
                    !type.getElementType().isBF16() ||
                    !IsVector4(args[1], true) || !IsI32(args[2]))
                    return Error(call, ctx,
                                 "raw_buffer_atomic_add_bf16x2 expects "
                                 "vector<2xbf16>, vector<4xi32> and i32");
                return true;
            }};
}

SymbolScope::Function BufferStoreU8() {
    return {
        [](ast::Call *call, GeneratorContext *ctx, Args args) {
            auto &builder = ctx->GetCurrentFunctionGenerator()->GetBuilder();
            auto loc = ctx->GetMLIRLocation(builder.getContext(), call);
            mlir::LLVM::CallIntrinsicOp::create(
                builder, loc,
                builder.getStringAttr("llvm.amdgcn.raw.buffer.store"),
                mlir::ValueRange{
                    args[0], args[1], args[2], args[3],
                    Constant(builder, loc,
                             *ConstantFolder::FoldIntValue(args[4]))});
            return ctx->GetCurrentFunctionGenerator()
                ->GetExprGenerator()
                ->CreateVoidValue();
        },
        [](ast::Call *call, GeneratorContext *ctx, Args args) {
            if (!CheckArgs(call, args, 5) || !args[0].getType().isInteger(8) ||
                !IsVector4(args[1], true) || !IsI32(args[2]) || !IsI32(args[3]))
                return Error(call, ctx,
                             "raw_buffer_store_u8 expects i8 value, resource, "
                             "i32 offsets and constant aux");
            auto aux = ConstantFolder::FoldIntValue(args[4]);
            if (!aux || *aux < 0 || *aux > 31)
                return Error(
                    call, ctx,
                    "raw_buffer_store_u8 aux must be a constant in [0, 31]");
            return true;
        }};
}

SymbolScope::Function DsSwizzle() {
    return {
        [](ast::Call *call, GeneratorContext *ctx, Args args) {
            auto &builder = ctx->GetCurrentFunctionGenerator()->GetBuilder();
            auto loc = ctx->GetMLIRLocation(builder.getContext(), call);
            auto pattern =
                Constant(builder, loc, *ConstantFolder::FoldIntValue(args[1]));
            auto value =
                Intrinsic(builder, loc, builder.getI32Type(),
                          "llvm.amdgcn.ds.swizzle", {args[0], pattern});
            SetTypeInfo(value, GetTypeInfo(args[0]));
            return value;
        },
        [](ast::Call *call, GeneratorContext *ctx, Args args) {
            if (!CheckArgs(call, args, 2) || !IsI32(args[0]))
                return Error(
                    call, ctx,
                    "ds_swizzle expects i32 value and constant 16-bit pattern");
            auto pattern = ConstantFolder::FoldIntValue(args[1]);
            if (!pattern || *pattern < 0 || *pattern > 65535)
                return Error(
                    call, ctx,
                    "ds_swizzle pattern must be a constant in [0, 65535]");
            return true;
        }};
}

SymbolScope::Function MaximumF32() {
    return {
        [](ast::Call *call, GeneratorContext *ctx, Args args) {
            auto &builder = ctx->GetCurrentFunctionGenerator()->GetBuilder();
            auto loc = ctx->GetMLIRLocation(builder.getContext(), call);
            return Intrinsic(builder, loc, builder.getF32Type(), "llvm.maximum",
                             args);
        },
        [](ast::Call *call, GeneratorContext *ctx, Args args) {
            if (!CheckArgs(call, args, 2) || !args[0].getType().isF32() ||
                !args[1].getType().isF32())
                return Error(call, ctx, "maximum_f32 expects two f32 values");
            return true;
        }};
}

} // namespace causalflow::avelang::ir::intrinsics
