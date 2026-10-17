#include "type_system.h"

#include "Dialect/AveLang/IR/AveLangOps.h"

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wambiguous-reversed-operator"
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Vector/IR/VectorOps.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#pragma clang diagnostic pop

namespace causalflow::avelang::ir {

namespace cf = causalflow::avelang::dialect;

namespace {

inline constexpr llvm::StringLiteral kValueTypeInfoAttr =
    "ave.type_info.is_unsigned_integer";
inline constexpr llvm::StringLiteral kMemRefTypeInfoAttr =
    "ave.type_info.element_is_unsigned_integer";
enum class IntegerSignednessAttr : int32_t {
    kSigned = 1,
    kUnsigned = 2,
};

TypeInfo DefaultTypeInfo(mlir::Type type) {
    TypeInfo info;
    if (IsIntegerValueOrVectorOfIntegers(type) ||
        IsMemRefWithIntegerElements(type)) {
        info.is_unsigned_integer = false;
    }
    return info;
}

TypeInfo TypeInfoFromAttr(std::optional<bool> value) {
    TypeInfo info;
    info.is_unsigned_integer = value;
    return info;
}

std::optional<bool> DecodeTypeInfoAttr(mlir::IntegerAttr attr) {
    if (!attr) {
        return std::nullopt;
    }
    switch (static_cast<IntegerSignednessAttr>(attr.getInt())) {
    case IntegerSignednessAttr::kSigned:
        return false;
    case IntegerSignednessAttr::kUnsigned:
        return true;
    }
    return std::nullopt;
}

mlir::IntegerAttr EncodeTypeInfoAttr(mlir::MLIRContext *ctx, bool isUnsigned) {
    return mlir::IntegerAttr::get(
        mlir::IntegerType::get(ctx, 32),
        static_cast<int32_t>(isUnsigned ? IntegerSignednessAttr::kUnsigned
                                        : IntegerSignednessAttr::kSigned));
}

std::optional<bool> GetTypeInfoAttr(mlir::Operation *op,
                                    llvm::StringRef attrName) {
    if (!op) {
        return std::nullopt;
    }
    return DecodeTypeInfoAttr(op->getAttrOfType<mlir::IntegerAttr>(attrName));
}

std::optional<bool> GetFunctionArgTypeInfoAttr(mlir::BlockArgument arg,
                                               llvm::StringRef attrName) {
    auto *parentOp = arg.getOwner() ? arg.getOwner()->getParentOp() : nullptr;
    auto func = llvm::dyn_cast_or_null<mlir::func::FuncOp>(parentOp);
    if (!func) {
        return std::nullopt;
    }
    return DecodeTypeInfoAttr(
        func.getArgAttrOfType<mlir::IntegerAttr>(arg.getArgNumber(), attrName));
}

TypeInfo GetMemRefTypeInfo(mlir::Value value);

TypeInfo GetValueTypeInfo(mlir::Value value) {
    if (!value || !IsIntegerValueOrVectorOfIntegers(value.getType())) {
        return {};
    }

    if (auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(value)) {
        if (auto attr =
                GetFunctionArgTypeInfoAttr(blockArg, kValueTypeInfoAttr)) {
            return TypeInfoFromAttr(attr);
        }
        return DefaultTypeInfo(value.getType());
    }

    auto *defOp = value.getDefiningOp();
    if (!defOp) {
        return DefaultTypeInfo(value.getType());
    }

    if (auto attr = GetTypeInfoAttr(defOp, kValueTypeInfoAttr)) {
        return TypeInfoFromAttr(attr);
    }

    if (auto load = llvm::dyn_cast<cf::AveLangMemRefLoadOp>(defOp)) {
        return GetTypeInfo(load.getMemref());
    }
    if (auto loadVec = llvm::dyn_cast<cf::AveLangMemRefLoadVecOp>(defOp)) {
        return GetTypeInfo(loadVec.getMemref());
    }
    if (auto extract = llvm::dyn_cast<mlir::vector::ExtractOp>(defOp)) {
        return GetTypeInfo(extract.getSource());
    }
    if (auto shapeCast = llvm::dyn_cast<mlir::vector::ShapeCastOp>(defOp)) {
        return GetTypeInfo(shapeCast.getSource());
    }
    if (auto bitcast = llvm::dyn_cast<mlir::vector::BitCastOp>(defOp)) {
        return GetTypeInfo(bitcast.getSource());
    }
    if (auto indexCast = llvm::dyn_cast<mlir::arith::IndexCastOp>(defOp)) {
        return GetTypeInfo(indexCast.getIn());
    }
    if (auto extui = llvm::dyn_cast<mlir::arith::ExtUIOp>(defOp)) {
        return TypeInfoFromAttr(true);
    }
    if (auto extsi = llvm::dyn_cast<mlir::arith::ExtSIOp>(defOp)) {
        return TypeInfoFromAttr(false);
    }
    if (auto trunc = llvm::dyn_cast<mlir::arith::TruncIOp>(defOp)) {
        return GetTypeInfo(trunc.getIn());
    }
    if (auto bitcast = llvm::dyn_cast<mlir::arith::BitcastOp>(defOp)) {
        return GetTypeInfo(bitcast.getIn());
    }
    if (auto cast = llvm::dyn_cast<mlir::UnrealizedConversionCastOp>(defOp)) {
        if (cast.getNumOperands() == 1) {
            return GetTypeInfo(cast.getOperand(0));
        }
    }
    if (auto select = llvm::dyn_cast<mlir::arith::SelectOp>(defOp)) {
        auto trueInfo = GetTypeInfo(select.getTrueValue());
        auto falseInfo = GetTypeInfo(select.getFalseValue());
        if (trueInfo.is_unsigned_integer == falseInfo.is_unsigned_integer) {
            return trueInfo;
        }
    }

    return DefaultTypeInfo(value.getType());
}

TypeInfo GetMemRefTypeInfo(mlir::Value value) {
    if (!value || !IsMemRefWithIntegerElements(value.getType())) {
        return {};
    }

    if (auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(value)) {
        if (auto attr =
                GetFunctionArgTypeInfoAttr(blockArg, kMemRefTypeInfoAttr)) {
            return TypeInfoFromAttr(attr);
        }
        return DefaultTypeInfo(value.getType());
    }

    auto *defOp = value.getDefiningOp();
    if (!defOp) {
        return DefaultTypeInfo(value.getType());
    }

    if (auto attr = GetTypeInfoAttr(defOp, kMemRefTypeInfoAttr)) {
        return TypeInfoFromAttr(attr);
    }

    if (auto subview = llvm::dyn_cast<cf::AveLangMemRefSubViewOp>(defOp)) {
        return GetTypeInfo(subview.getSource());
    }
    if (auto cast = llvm::dyn_cast<cf::AveLangMemRefCastOp>(defOp)) {
        return GetTypeInfo(cast.getSource());
    }
    if (auto full = llvm::dyn_cast<cf::FullOp>(defOp)) {
        return GetTypeInfo(full.getValue());
    }

    return DefaultTypeInfo(value.getType());
}

} // namespace

bool IsIntegerValueOrVectorOfIntegers(mlir::Type type) {
    if (type.isInteger()) {
        return true;
    }

    auto vectorType = mlir::dyn_cast<mlir::VectorType>(type);
    return vectorType && vectorType.getElementType().isInteger();
}

bool IsMemRefWithIntegerElements(mlir::Type type) {
    auto memrefType = mlir::dyn_cast<cf::MemRefType>(type);
    return memrefType &&
           IsIntegerValueOrVectorOfIntegers(memrefType.getElementType());
}

TypeInfo GetTypeInfo(mlir::Value value) {
    if (!value) {
        return {};
    }
    if (IsMemRefWithIntegerElements(value.getType())) {
        return GetMemRefTypeInfo(value);
    }
    return GetValueTypeInfo(value);
}

void SetTypeInfo(mlir::Value value, const TypeInfo &typeInfo) {
    if (!value || !typeInfo.is_unsigned_integer) {
        return;
    }

    auto setAttr = [&](llvm::StringRef attrName) {
        auto attr = EncodeTypeInfoAttr(value.getContext(),
                                       *typeInfo.is_unsigned_integer);
        if (auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(value)) {
            auto *parentOp = blockArg.getOwner()
                                 ? blockArg.getOwner()->getParentOp()
                                 : nullptr;
            if (auto func =
                    llvm::dyn_cast_or_null<mlir::func::FuncOp>(parentOp)) {
                func.setArgAttr(blockArg.getArgNumber(), attrName, attr);
            }
            return;
        }
        if (auto *defOp = value.getDefiningOp()) {
            defOp->setAttr(attrName, attr);
        }
    };

    if (IsMemRefWithIntegerElements(value.getType())) {
        setAttr(kMemRefTypeInfoAttr);
    } else if (IsIntegerValueOrVectorOfIntegers(value.getType())) {
        setAttr(kValueTypeInfoAttr);
    }
}

} // namespace causalflow::avelang::ir
