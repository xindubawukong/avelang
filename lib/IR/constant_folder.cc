#include "constant_folder.h"

#include "AST/ast_nodes_expr.h"
#include "Dialect/AveLang/IR/AveLangOps.h"
#include "generator_context.h"
#include "parsing_utils.h"
#include "symbol_table.h"
#include "type_system.h"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/BuiltinAttributes.h>

#include <llvm/Support/Casting.h>

namespace causalflow::avelang::ir {

namespace {

static std::optional<int64_t> GetStaticElementCount(mlir::Type type) {
    auto getCount = [](auto memrefType) -> std::optional<int64_t> {
        int64_t count = 1;
        for (int64_t dim : memrefType.getShape()) {
            if (dim == mlir::ShapedType::kDynamic) {
                return std::nullopt;
            }
            count *= dim;
        }
        return count;
    };

    if (auto avelangType =
            mlir::dyn_cast<causalflow::avelang::dialect::MemRefType>(type)) {
        return getCount(avelangType);
    }
    if (auto memrefType = mlir::dyn_cast<mlir::MemRefType>(type)) {
        return getCount(memrefType);
    }
    return std::nullopt;
}

static bool AreZeroIndices(mlir::ValueRange indices) {
    return llvm::all_of(indices, [](mlir::Value index) {
        if (auto constOp = index.getDefiningOp<mlir::arith::ConstantOp>()) {
            if (auto intAttr =
                    mlir::dyn_cast<mlir::IntegerAttr>(constOp.getValue())) {
                return intAttr.getInt() == 0;
            }
        }
        if (auto constIndex =
                index.getDefiningOp<mlir::arith::ConstantIndexOp>()) {
            return constIndex.value() == 0;
        }
        return false;
    });
}

static std::optional<int64_t> FoldScalarMemrefValue(mlir::Value memref) {
    auto elementCount = GetStaticElementCount(memref.getType());
    if (!elementCount || *elementCount != 1) {
        return std::nullopt;
    }

    std::optional<int64_t> foldedValue;
    for (mlir::Operation *user : memref.getUsers()) {
        if (auto store = mlir::dyn_cast<
                causalflow::avelang::dialect::AveLangMemRefStoreOp>(user)) {
            if (!AreZeroIndices(store.getIndices())) {
                return std::nullopt;
            }
            auto value = ConstantFolder::FoldIntValue(store.getValue());
            if (!value || foldedValue) {
                return std::nullopt;
            }
            foldedValue = value;
            continue;
        }

        if (auto store = mlir::dyn_cast<mlir::memref::StoreOp>(user)) {
            if (!AreZeroIndices(store.getIndices())) {
                return std::nullopt;
            }
            auto value = ConstantFolder::FoldIntValue(store.getValue());
            if (!value || foldedValue) {
                return std::nullopt;
            }
            foldedValue = value;
            continue;
        }

        if (auto load = mlir::dyn_cast<
                causalflow::avelang::dialect::AveLangMemRefLoadOp>(user)) {
            if (!AreZeroIndices(load.getIndices())) {
                return std::nullopt;
            }
            continue;
        }

        if (auto load = mlir::dyn_cast<mlir::memref::LoadOp>(user)) {
            if (!AreZeroIndices(load.getIndices())) {
                return std::nullopt;
            }
            continue;
        }

        return std::nullopt;
    }

    return foldedValue;
}

template <typename LoadOpTy>
static std::optional<int64_t> FoldScalarMemrefLoadImpl(LoadOpTy load) {
    auto memref = load.getMemref();
    auto elementCount = GetStaticElementCount(memref.getType());
    if (!elementCount || *elementCount != 1 ||
        !AreZeroIndices(load.getIndices())) {
        return std::nullopt;
    }

    std::optional<int64_t> foldedValue;
    for (mlir::Operation *user : memref.getUsers()) {
        if (user == load.getOperation()) {
            continue;
        }

        if (auto store = mlir::dyn_cast<
                causalflow::avelang::dialect::AveLangMemRefStoreOp>(user)) {
            if (!AreZeroIndices(store.getIndices())) {
                return std::nullopt;
            }
            auto value = ConstantFolder::FoldIntValue(store.getValue());
            if (!value || foldedValue) {
                return std::nullopt;
            }
            foldedValue = value;
            continue;
        }

        if (auto store = mlir::dyn_cast<mlir::memref::StoreOp>(user)) {
            if (!AreZeroIndices(store.getIndices())) {
                return std::nullopt;
            }
            auto value = ConstantFolder::FoldIntValue(store.getValue());
            if (!value || foldedValue) {
                return std::nullopt;
            }
            foldedValue = value;
            continue;
        }

        return std::nullopt;
    }

    return foldedValue;
}

// Keep widths until the public shape/index boundary. APInt gives the same
// wrapping arithmetic as arith operations, without host signed-overflow UB.
struct FoldedInteger {
    llvm::APInt bits;
    bool isUnsigned = false;

    int64_t getInt() const {
        return isUnsigned || bits.getBitWidth() == 1
                   ? static_cast<int64_t>(bits.getZExtValue())
                   : bits.getSExtValue();
    }

    llvm::APInt extend(unsigned width, bool zeroExtend = false) const {
        return zeroExtend || isUnsigned || bits.getBitWidth() == 1
                   ? bits.zextOrTrunc(width)
                   : bits.sextOrTrunc(width);
    }
};

static bool IsNamedCall(const ast::Call *call, llvm::StringRef name) {
    if (auto *func = llvm::dyn_cast<ast::Name>(call->GetFunc()))
        return func->GetId() == name;
    if (auto *func = llvm::dyn_cast<ast::AttributeExpr>(call->GetFunc()))
        return func->GetAttr() == name;
    return false;
}

static std::optional<FoldedInteger> EvaluateInteger(ast::Expr *expr,
                                                    GeneratorContext *ctx) {
    if (!expr)
        return std::nullopt;
    if (auto *constant = llvm::dyn_cast<ast::Constant>(expr)) {
        if (constant->GetValue() == "True" || constant->GetValue() == "False")
            return FoldedInteger{
                llvm::APInt(1, constant->GetValue() == "True")};
        if (auto value = ParseConstantInteger(constant->GetValue())) {
            // Shape literals may be i64; runtime literals are checked
            // separately.
            unsigned width = llvm::isInt<32>(*value) ? 32 : 64;
            return FoldedInteger{llvm::APInt(width, *value)};
        }
        return std::nullopt;
    }
    if (llvm::isa<ast::Name, ast::AttributeExpr>(expr) && ctx && ctx->syms) {
        auto symbol = ctx->syms->ResolveSymbol(expr, std::nullopt,
                                               /*report_not_found=*/false);
        if (!symbol || !symbol->isa(SymbolTable::SymbolKind::kValue))
            return std::nullopt;
        auto value = symbol->value;
        auto folded = ConstantFolder::FoldIntValue(value);
        auto type = value.getType();
        if (!folded) {
            folded = FoldScalarMemrefValue(value);
            if (auto memref = mlir::dyn_cast<dialect::MemRefType>(type))
                type = memref.getElementType();
            else if (auto memref = mlir::dyn_cast<mlir::MemRefType>(type))
                type = memref.getElementType();
        }
        if (!folded || !type.isIntOrIndex())
            return std::nullopt;
        unsigned width = type.isIndex() ? 64 : type.getIntOrFloatBitWidth();
        return FoldedInteger{
            llvm::APInt(width, *folded),
            GetTypeInfo(value).is_unsigned_integer.value_or(false)};
    }
    if (auto *call = llvm::dyn_cast<ast::Call>(expr)) {
        bool bitcast = IsNamedCall(call, "bitcast");
        if ((!bitcast && !IsNamedCall(call, "convert")) ||
            call->GetArgs().size() != 2 || !ctx || !ctx->syms)
            return std::nullopt;
        auto value = EvaluateInteger(call->GetArgs()[0], ctx);
        auto target = ctx->syms->ResolveTypeWithInfo(call->GetArgs()[1]);
        auto type = mlir::dyn_cast_or_null<mlir::IntegerType>(target.type);
        if (!value || !type || !target.info.is_unsigned_integer ||
            (bitcast && value->bits.getBitWidth() != type.getWidth()))
            return std::nullopt;
        return FoldedInteger{value->extend(type.getWidth()),
                             *target.info.is_unsigned_integer};
    }
    if (auto *unary = llvm::dyn_cast<ast::UnaryOp>(expr)) {
        auto operand = EvaluateInteger(unary->GetOperand(), ctx);
        if (!operand)
            return std::nullopt;
        if (unary->GetOp() == "UAdd")
            return operand;
        if (unary->GetOp() == "USub") {
            operand->bits.negate();
            // Match the runtime parser's special case for the i32 minimum.
            if (auto *literal =
                    llvm::dyn_cast<ast::Constant>(unary->GetOperand());
                literal && ParseConstantInteger(literal->GetValue()) ==
                               int64_t{2147483648})
                operand->bits = operand->bits.trunc(32);
            return operand;
        }
        if (unary->GetOp() == "Invert") {
            operand->bits.flipAllBits();
            return operand;
        }
        return std::nullopt;
    }
    auto *binop = llvm::dyn_cast<ast::BinOp>(expr);
    if (!binop)
        return std::nullopt;
    auto lhs = EvaluateInteger(binop->GetLeft(), ctx);
    auto rhs = EvaluateInteger(binop->GetRight(), ctx);
    if (!lhs || !rhs)
        return std::nullopt;
    const auto &op = binop->GetOp();
    bool shift = op == "LShift" || op == "RShift";
    bool bitwise = shift || op == "BitAnd" || op == "BitOr" || op == "BitXor";
    unsigned width = std::max(lhs->bits.getBitWidth(), rhs->bits.getBitWidth());
    bool isUnsigned =
        shift ? lhs->isUnsigned : lhs->isUnsigned || rhs->isUnsigned;
    auto a = lhs->extend(width, bitwise);
    auto b = rhs->extend(width, bitwise);
    auto result = [&](llvm::APInt bits) -> std::optional<FoldedInteger> {
        return FoldedInteger{std::move(bits), isUnsigned};
    };
    if (op == "Add")
        return result(a + b);
    if (op == "Sub")
        return result(a - b);
    if (op == "Mult")
        return result(a * b);
    if (op == "BitAnd")
        return result(a & b);
    if (op == "BitOr")
        return result(a | b);
    if (op == "BitXor")
        return result(a ^ b);
    if (shift) {
        // Oversized (including negative) shifts produce poison at runtime.
        if (b.uge(width))
            return std::nullopt;
        if (op == "LShift")
            return result(a.shl(b));
        return result(isUnsigned ? a.lshr(b) : a.ashr(b));
    }
    if (op != "Div" && op != "FloorDiv" && op != "Mod")
        return std::nullopt;
    if (b.isZero() || (!isUnsigned && a.isMinSignedValue() && b.isAllOnes()))
        return std::nullopt;
    if (isUnsigned)
        return result(op == "Mod" ? a.urem(b) : a.udiv(b));
    if (op == "Mod")
        return result(a.srem(b));
    if (op == "FloorDiv" && !a.srem(b).isZero()) {
        // Mirror the runtime floor-division expansion, including its widths.
        auto adjustment = a.srem(b) + b;
        if (adjustment.isMinSignedValue() && b.isAllOnes())
            return std::nullopt;
        a -= adjustment.srem(b);
        if (a.isMinSignedValue() && b.isAllOnes())
            return std::nullopt;
    }
    return result(a.sdiv(b));
}

// Delegate arithmetic to MLIR's own folders so widths, casts, unsigned
// predicates and poison cases stay in sync with the emitted instructions.
static std::optional<llvm::APInt> FoldIntegerBits(mlir::Value value) {
    if (!value || !value.getType().isIntOrIndex())
        return std::nullopt;
    auto *op = value.getDefiningOp();
    if (!op)
        return std::nullopt;
    if (auto constant = mlir::dyn_cast<mlir::arith::ConstantOp>(op)) {
        if (auto attr = mlir::dyn_cast<mlir::IntegerAttr>(constant.getValue()))
            return attr.getValue();
    }
    if (auto cast = mlir::dyn_cast<mlir::UnrealizedConversionCastOp>(op)) {
        if (cast.getNumOperands() == 1 && cast.getNumResults() == 1 &&
            cast.getOperand(0).getType() == value.getType())
            return FoldIntegerBits(cast.getOperand(0));
        return std::nullopt;
    }
    std::optional<int64_t> loaded;
    if (auto load = mlir::dyn_cast<dialect::AveLangMemRefLoadOp>(op))
        loaded = FoldScalarMemrefLoadImpl(load);
    else if (auto load = mlir::dyn_cast<mlir::memref::LoadOp>(op))
        loaded = FoldScalarMemrefLoadImpl(load);
    if (loaded) {
        unsigned width = value.getType().isIndex()
                             ? 64
                             : value.getType().getIntOrFloatBitWidth();
        return llvm::APInt(width, *loaded);
    }
    if (auto select = mlir::dyn_cast<mlir::arith::SelectOp>(op)) {
        if (auto cond = ConstantFolder::FoldBoolValue(select.getCondition()))
            return FoldIntegerBits(*cond ? select.getTrueValue()
                                         : select.getFalseValue());
        return std::nullopt;
    }
    if (op->getName().getDialectNamespace() != "arith" ||
        op->getNumResults() != 1)
        return std::nullopt;
    llvm::SmallVector<mlir::Attribute> operands;
    for (auto operand : op->getOperands()) {
        auto bits = FoldIntegerBits(operand);
        if (!bits)
            return std::nullopt;
        operands.push_back(mlir::IntegerAttr::get(operand.getType(), *bits));
    }
    // Folders may also perform in-place canonicalizations. Fold a temporary
    // operation with abstract operands so this query never mutates source IR.
    mlir::Block scratch;
    for (auto operand : op->getOperands())
        scratch.addArgument(operand.getType(), op->getLoc());
    auto *copy = op->clone();
    scratch.push_back(copy);
    copy->setOperands(scratch.getArguments());
    llvm::SmallVector<mlir::OpFoldResult> results;
    if (mlir::failed(copy->fold(operands, results)) || results.size() != 1)
        return std::nullopt;
    if (auto attr = mlir::dyn_cast_if_present<mlir::IntegerAttr>(
            results[0].dyn_cast<mlir::Attribute>()))
        return attr.getValue();
    if (auto folded = mlir::dyn_cast_if_present<mlir::BlockArgument>(
            results[0].dyn_cast<mlir::Value>());
        folded && folded.getOwner() == &scratch)
        return mlir::cast<mlir::IntegerAttr>(operands[folded.getArgNumber()])
            .getValue();
    return std::nullopt;
}

} // namespace

std::optional<bool> ConstantFolder::FoldBoolValue(mlir::Value value) {
    if (auto bits = FoldIntegerBits(value))
        return !bits->isZero();
    return std::nullopt;
}

std::optional<int64_t> ConstantFolder::FoldIntValue(mlir::Value value) {
    if (auto bits = FoldIntegerBits(value))
        return FoldedInteger{
            *bits, GetTypeInfo(value).is_unsigned_integer.value_or(false)}
            .getInt();
    return std::nullopt;
}

std::optional<int64_t> ConstantFolder::Evaluate(ast::Expr *expr) const {
    if (auto value = EvaluateInteger(expr, ctx_))
        return value->getInt();
    return std::nullopt;
}

} // namespace causalflow::avelang::ir
