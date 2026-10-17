#include "layout_operation.h"
#include "Dialect/AveLang/IR/AveLangOps.h"
#include "Utils/assert.h"
#include "constant_folder.h"
#include "generator_context.h"
#include "mlir_generator_impl.h"
#include "parsing_utils.h"
#include "type_promotion.h"
#include "type_system.h"

#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>

namespace causalflow::avelang::ir {

using namespace mlir;
namespace cf = causalflow::avelang::dialect;

static cf::MemRefType
createAveLangMemRefType(mlir::MLIRContext *context,
                        llvm::ArrayRef<int64_t> shape,
                        llvm::ArrayRef<int64_t> strides, mlir::Type elementType,
                        mlir::Attribute memorySpace = {}) {
    auto layoutType = cf::LayoutType::get(context, shape, strides);
    return cf::MemRefType::get(context, layoutType, elementType, memorySpace);
}

long LayoutOperation::extractIndexValue(mlir::Value value) {
    if (auto folded = ConstantFolder::FoldIntValue(value)) {
        return *folded;
    }
    return mlir::ShapedType::kDynamic;
}

// Extract values from potentially nested tuples (flattens the tuple)
bool LayoutOperation::flattenTupleValues(mlir::Value tupleValue,
                                         llvm::SmallVector<int64_t> &values) {
    if (auto makeTupleOp = tupleValue.getDefiningOp<cf::MakeIntTupleOp>()) {
        for (auto elem : makeTupleOp.getElements()) {
            // Recursively handle nested tuples
            if (elem.getDefiningOp<cf::MakeIntTupleOp>()) {
                if (!flattenTupleValues(elem, values)) {
                    return false;
                }
            } else {
                values.push_back(extractIndexValue(elem));
            }
        }
        return true;
    }

    // If it's not a MakeIntTupleOp, treat it as a single value
    values.push_back(extractIndexValue(tupleValue));
    return true;
}

mlir::Value LayoutOperation::CastToIndex(mlir::OpBuilder &builder,
                                         mlir::Location loc,
                                         mlir::Value value) {
    if (value.getType().isIndex()) {
        return value;
    }
    if (value.getType().isIntOrIndexOrFloat()) {
        return CreateIndexCast(builder, loc, builder.getIndexType(), value);
    }
    return value;
}

void LayoutOperation::CollectTupleLeaves(mlir::Value value,
                                         llvm::SmallVector<mlir::Value> &out) {
    if (auto tupleOp = value.getDefiningOp<cf::MakeIntTupleOp>()) {
        for (auto elem : tupleOp.getElements()) {
            CollectTupleLeaves(elem, out);
        }
        return;
    }
    out.push_back(value);
}

void LayoutOperation::ExpandLinearIndex(mlir::OpBuilder &builder,
                                        mlir::Location loc,
                                        mlir::Value linearIndex,
                                        llvm::ArrayRef<mlir::Value> dims,
                                        llvm::SmallVector<mlir::Value> &out) {
    if (dims.empty()) {
        return;
    }

    auto linear = CastToIndex(builder, loc, linearIndex);
    llvm::SmallVector<mlir::Value> basis;
    if (dims.size() > 1) {
        basis.reserve(dims.size() - 1);
        for (auto it = dims.rbegin(); it != dims.rend(); ++it) {
            if (it == dims.rbegin()) {
                continue;
            }
            basis.push_back(CastToIndex(builder, loc, *it));
        }
    }

    auto delinearize = mlir::affine::AffineDelinearizeIndexOp::create(
        builder, loc, linear, basis, /*hasOuterBound=*/false);
    auto results = delinearize.getResults();
    for (auto index = results.size(); index > 0; --index) {
        out.push_back(results[index - 1]);
    }
}

bool LayoutOperation::ExpandIndicesForNestedLayout(
    mlir::OpBuilder &builder, mlir::Value memrefValue,
    llvm::SmallVector<mlir::Value> &indices) {
    auto memrefType = mlir::dyn_cast<cf::MemRefType>(memrefValue.getType());
    if (!memrefType || indices.empty()) {
        return false;
    }

    auto castOp = memrefValue.getDefiningOp<cf::AveLangMemRefCastOp>();
    if (!castOp || !castOp.getLayout()) {
        return false;
    }

    auto layoutOp = castOp.getLayout().getDefiningOp<cf::MakeLayoutOp>();
    if (!layoutOp) {
        return false;
    }

    auto dimsTuple = layoutOp.getDims().getDefiningOp<cf::MakeIntTupleOp>();
    if (!dimsTuple) {
        return false;
    }

    bool hasNested = false;
    for (auto elem : dimsTuple.getElements()) {
        if (elem.getDefiningOp<cf::MakeIntTupleOp>()) {
            hasNested = true;
            break;
        }
    }
    if (!hasNested) {
        return false;
    }

    size_t logicalRank = dimsTuple.getNumElements();
    if (indices.size() > logicalRank) {
        return false;
    }

    if (logicalRank == static_cast<size_t>(memrefType.getRank())) {
        return false;
    }

    mlir::Location loc = builder.getUnknownLoc();
    llvm::SmallVector<mlir::Value> expanded;
    expanded.reserve(indices.size());

    size_t idxPos = 0;
    for (auto elem : dimsTuple.getElements()) {
        if (idxPos >= indices.size()) {
            break;
        }
        auto idxVal = indices[idxPos++];
        if (elem.getDefiningOp<cf::MakeIntTupleOp>()) {
            llvm::SmallVector<mlir::Value> nestedDims;
            CollectTupleLeaves(elem, nestedDims);
            ExpandLinearIndex(builder, loc, idxVal, nestedDims, expanded);
        } else {
            expanded.push_back(idxVal);
        }
    }

    if (expanded.size() == indices.size()) {
        return false;
    }

    indices.swap(expanded);
    return true;
}

mlir::Value
LayoutOperation::createViewFunction(ast::Call *callExpr, GeneratorContext *ctx,
                                    llvm::ArrayRef<mlir::Value> resolvedArgs) {
    auto location =
        ctx->GetCurrentFunctionGenerator()->GetBuilder().getUnknownLoc();
    auto &builder = ctx->GetCurrentFunctionGenerator()->GetBuilder();

    const auto &args = callExpr->GetArgs();

    // Validate argument count - view(memref, dtype, layout)
    if (args.size() != 3) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        callExpr->GetSourceRange().getBegin())
            << "view() expects exactly 3 arguments: memref, dtype, layout";
        return nullptr;
    }

    if (resolvedArgs.size() < 3) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        callExpr->GetSourceRange().getBegin())
            << "Failed to resolve arguments for view()";
        return nullptr;
    }

    // Get the memref to cast
    auto memrefValue = resolvedArgs[0];
    if (!memrefValue) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        callExpr->GetSourceRange().getBegin())
            << "Failed to generate memref argument for view()";
        return nullptr;
    }

    // Verify it's a memref type
    auto memrefType = mlir::dyn_cast<cf::MemRefType>(memrefValue.getType());
    if (!memrefType) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        callExpr->GetSourceRange().getBegin())
            << "First argument to view() must be a memref";
        return nullptr;
    }

    // Resolve the element type from the second argument
    auto element = ctx->syms->ResolveBuiltinType(args[1]);
    auto elementType = element.type;
    if (!elementType || !elementType.isSignlessIntOrFloat()) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        callExpr->GetSourceRange().getBegin())
            << "Failed to resolve element type for view()";
        return nullptr;
    }

    // Get the layout from the third argument
    auto layoutValue = resolvedArgs[2];
    if (!layoutValue) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        callExpr->GetSourceRange().getBegin())
            << "Failed to resolve layout argument for view()";
        return nullptr;
    }

    // Check if layout is created by make_layout.
    auto makeLayoutOp = layoutValue.getDefiningOp<cf::MakeLayoutOp>();
    if (!makeLayoutOp) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        callExpr->GetSourceRange().getBegin())
            << "Third argument to view() must be a layout created by "
               "make_layout";
        return nullptr;
    }

    // Get dims and stride from the layout
    auto dimsValue = makeLayoutOp.getDims();
    auto strideValue = makeLayoutOp.getStride();

    auto dimsTupleOp = dimsValue.getDefiningOp<cf::MakeIntTupleOp>();
    auto strideTupleOp = strideValue.getDefiningOp<cf::MakeIntTupleOp>();

    if (!dimsTupleOp || !strideTupleOp) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        callExpr->GetSourceRange().getBegin())
            << "Internal error: layout must contain valid dims and stride "
               "tuples";
        return nullptr;
    }

    // Validate that dims and stride have the same dimensions
    if (dimsTupleOp.getNumElements() != strideTupleOp.getNumElements()) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        callExpr->GetSourceRange().getBegin())
            << "Layout dims and stride must have the same number of dimensions";
        return nullptr;
    }

    if (dimsTupleOp.getNumElements() == 0) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        callExpr->GetSourceRange().getBegin())
            << "Layout dims and stride cannot be empty";
        return nullptr;
    }

    // Process shapes to build static shape array
    llvm::SmallVector<int64_t> staticShape;
    llvm::SmallVector<int64_t> staticStrides;
    flattenTupleValues(dimsValue, staticShape); // flatten tuple
    flattenTupleValues(strideValue, staticStrides);
    if (staticShape.size() != staticStrides.size()) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        callExpr->GetSourceRange().getBegin())
            << "Layout dims and stride must have the same number of dimensions";
        return nullptr;
    }

    // Create the new memref type with the specified shape and element type
    auto newMemRefType = createAveLangMemRefType(
        builder.getContext(), staticShape, staticStrides, elementType,
        memrefType.getMemorySpace());

    // Create the new memref.cast op from the AveLang dialect
    auto castOp = cf::AveLangMemRefCastOp::create(
        builder, location, memrefValue, layoutValue, newMemRefType);
    if (auto typeInfo = element.info; typeInfo.is_unsigned_integer) {
        SetTypeInfo(castOp.getResult(), typeInfo);
    } else {
        SetTypeInfo(castOp.getResult(), GetTypeInfo(memrefValue));
    }

    return castOp.getResult();
}

mlir::Value LayoutOperation::createMakeLayoutFunction(
    ast::Call *callExpr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolvedArgs) {
    auto location =
        ctx->GetCurrentFunctionGenerator()->GetBuilder().getUnknownLoc();
    auto &builder = ctx->GetCurrentFunctionGenerator()->GetBuilder();

    const auto &args = callExpr->GetArgs();

    // Validate argument count - make_layout(dims, stride)
    if (args.size() != 2) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        callExpr->GetSourceRange().getBegin())
            << "make_layout() expects exactly 2 arguments: dims and stride";
        return nullptr;
    }

    if (resolvedArgs.size() < 2 || !resolvedArgs[0] || !resolvedArgs[1]) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        callExpr->GetSourceRange().getBegin())
            << "Failed to resolve arguments for make_layout()";
        return nullptr;
    }

    auto dimsValue = resolvedArgs[0];
    auto strideValue = resolvedArgs[1];

    // Get the dims and stride as tuples
    auto dimsTupleOp = dimsValue.getDefiningOp<cf::MakeIntTupleOp>();
    auto strideTupleOp = strideValue.getDefiningOp<cf::MakeIntTupleOp>();

    if (!dimsTupleOp || !strideTupleOp) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        callExpr->GetSourceRange().getBegin())
            << "Dims and stride arguments to make_layout() must be integer "
               "tuples";
        return nullptr;
    }

    // Validate that dims and stride have the same number of elements
    if (dimsTupleOp.getNumElements() != strideTupleOp.getNumElements()) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        callExpr->GetSourceRange().getBegin())
            << "Dims and stride tuples must have the same number of elements";
        return nullptr;
    }

    if (dimsTupleOp.getNumElements() == 0) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        callExpr->GetSourceRange().getBegin())
            << "Dims and stride tuples cannot be empty";
        return nullptr;
    }

    // Normalize while TypeInfo is still available. Lowering can then consume
    // index leaves directly without guessing signedness from signless MLIR.
    auto toIndexTuple = [&](auto &&self, mlir::Value value) -> mlir::Value {
        if (auto tuple = value.getDefiningOp<cf::MakeIntTupleOp>()) {
            llvm::SmallVector<mlir::Value> elements;
            for (auto element : tuple.getElements()) {
                auto converted = self(self, element);
                if (!converted)
                    return nullptr;
                elements.push_back(converted);
            }
            return cf::MakeIntTupleOp::create(builder, location, elements);
        }
        if (!value.getType().isIntOrIndex())
            return nullptr;
        return CreateIndexCast(builder, location, builder.getIndexType(),
                               value);
    };
    dimsValue = toIndexTuple(toIndexTuple, dimsValue);
    strideValue = toIndexTuple(toIndexTuple, strideValue);
    if (!dimsValue || !strideValue) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        callExpr->GetSourceRange().getBegin())
            << "Dims and stride arguments to make_layout() must be integer "
               "tuples";
        return nullptr;
    }
    auto makeLayoutOp =
        cf::MakeLayoutOp::create(builder, location, dimsValue, strideValue);

    return makeLayoutOp.getResult();
}

} // namespace causalflow::avelang::ir
