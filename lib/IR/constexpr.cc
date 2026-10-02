#include "constexpr.h"

#include <llvm/Support/MathExtras.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/BuiltinAttributes.h>

namespace causalflow::avelang::ir {

llvm::Expected<mlir::Attribute> ParseConstexpr(const llvm::json::Object &info,
                                               mlir::Builder &builder) {
    auto type = info.getString("type");
    auto *value = info.get("value");
    if (!type || !value)
        return llvm::createStringError("constexpr requires type and value");

    if (*type == "i1") {
        if (auto v = value->getAsBoolean())
            return builder.getBoolAttr(*v);
    } else if (*type == "i32" || *type == "i64") {
        if (auto v = value->getAsInteger()) {
            if (*type == "i64")
                return builder.getI64IntegerAttr(*v);
            if (llvm::isInt<32>(*v))
                return builder.getI32IntegerAttr(*v);
        }
    } else if (*type == "f64") {
        if (auto v = value->getAsNumber())
            return builder.getF64FloatAttr(*v);
    } else if (*type == "dataclass") {
        auto *record = value->getAsObject();
        auto name = record ? record->getString("class") : std::nullopt;
        auto *fields = record ? record->getObject("fields") : nullptr;
        if (name && fields) {
            llvm::SmallVector<mlir::NamedAttribute> members;
            for (const auto &[key, item] : *fields) {
                auto *object = item.getAsObject();
                if (!object)
                    return llvm::createStringError(
                        "Invalid constexpr dataclass field");
                auto field = ParseConstexpr(*object, builder);
                if (!field)
                    return field.takeError();
                members.push_back(builder.getNamedAttr(key.str(), *field));
            }
            return builder.getDictionaryAttr({
                builder.getNamedAttr("class", builder.getStringAttr(*name)),
                builder.getNamedAttr("fields",
                                     builder.getDictionaryAttr(members)),
            });
        }
    }
    return llvm::createStringError("Invalid constexpr value for type '%s'",
                                   type->str().c_str());
}

mlir::Value MaterializeConstexpr(ConstexprValue value,
                                 mlir::OpBuilder &builder, mlir::Location loc) {
    if (!mlir::isa<mlir::IntegerAttr, mlir::FloatAttr>(value.value))
        return {};
    auto constant = mlir::arith::ConstantOp::create(
        builder, loc, mlir::cast<mlir::TypedAttr>(value.value));
    SetTypeInfo(constant, value.type_info);
    return constant;
}

} // namespace causalflow::avelang::ir
