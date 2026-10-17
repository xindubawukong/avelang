#include "type_promotion.h"

#include "type_system.h"

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wambiguous-reversed-operator"
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/BuiltinOps.h>
#pragma clang diagnostic pop

namespace causalflow::avelang::ir {

namespace {

mlir::Value CreateTypeInfoCast(mlir::OpBuilder &builder, mlir::Location loc,
                               mlir::Value value, const TypeInfo &typeInfo) {
    auto cast = mlir::UnrealizedConversionCastOp::create(
        builder, loc, mlir::TypeRange{value.getType()},
        mlir::ValueRange{value});
    SetTypeInfo(cast.getResult(0), typeInfo);
    return cast.getResult(0);
}

} // namespace

mlir::Value CreateTypeConversion(mlir::Value value, mlir::Type source_type,
                                 mlir::Type target_type,
                                 mlir::Location location,
                                 mlir::OpBuilder &builder, bool allow_demotion,
                                 std::optional<bool> source_unsigned,
                                 std::optional<bool> target_unsigned) {
    if (!source_unsigned) {
        source_unsigned = GetTypeInfo(value).is_unsigned_integer;
    }

    // If types are the same, only signedness may need retagging.
    if (source_type == target_type) {
        if (target_unsigned && target_unsigned != source_unsigned &&
            IsIntegerValueOrVectorOfIntegers(target_type)) {
            return CreateTypeInfoCast(builder, location, value,
                                      TypeInfo{*target_unsigned});
        }
        return value;
    }

    // Integer to integer conversions
    if (source_type.isInteger() && target_type.isInteger()) {
        auto source_width = source_type.getIntOrFloatBitWidth();
        auto target_width = target_type.getIntOrFloatBitWidth();

        if (source_width < target_width) {
            auto isUnsigned =
                source_type.isInteger(1) || source_unsigned.value_or(false);
            if (isUnsigned) {
                auto result = mlir::arith::ExtUIOp::create(builder, location,
                                                           target_type, value);
                SetTypeInfo(result.getResult(),
                            TypeInfo{target_unsigned.value_or(
                                source_unsigned.value_or(false))});
                return result;
            } else {
                auto result = mlir::arith::ExtSIOp::create(builder, location,
                                                           target_type, value);
                SetTypeInfo(result.getResult(),
                            TypeInfo{target_unsigned.value_or(
                                source_unsigned.value_or(false))});
                return result;
            }
        } else if (source_width > target_width) {
            if (allow_demotion) {
                auto result = mlir::arith::TruncIOp::create(builder, location,
                                                            target_type, value);
                SetTypeInfo(result.getResult(),
                            TypeInfo{target_unsigned.value_or(
                                source_unsigned.value_or(false))});
                return result;
            } else {
                return nullptr;
            }
        } else if (target_unsigned && source_unsigned &&
                   *target_unsigned != *source_unsigned) {
            return CreateTypeInfoCast(builder, location, value,
                                      TypeInfo{*target_unsigned});
        }
    }

    // Integer to float conversions
    if (source_type.isInteger() && mlir::isa<mlir::FloatType>(target_type)) {
        if (source_type.isInteger(1) || source_unsigned.value_or(false)) {
            return mlir::arith::UIToFPOp::create(builder, location, target_type,
                                                 value);
        } else {
            return mlir::arith::SIToFPOp::create(builder, location, target_type,
                                                 value);
        }
    }

    // Float to integer conversions
    if (mlir::isa<mlir::FloatType>(source_type) && target_type.isInteger()) {
        if (allow_demotion) {
            auto isUnsigned = target_unsigned.value_or(false);
            if (isUnsigned) {
                auto result = mlir::arith::FPToUIOp::create(builder, location,
                                                            target_type, value);
                SetTypeInfo(result.getResult(), TypeInfo{true});
                return result;
            } else {
                auto result = mlir::arith::FPToSIOp::create(builder, location,
                                                            target_type, value);
                SetTypeInfo(result.getResult(), TypeInfo{false});
                return result;
            }
        } else {
            return nullptr;
        }
    }

    // Float to float conversions
    if (mlir::isa<mlir::FloatType>(source_type) &&
        mlir::isa<mlir::FloatType>(target_type)) {
        auto source_width = source_type.getIntOrFloatBitWidth();
        auto target_width = target_type.getIntOrFloatBitWidth();

        if (source_width < target_width) {
            return mlir::arith::ExtFOp::create(builder, location, target_type,
                                               value);
        } else if (source_width > target_width) {
            if (allow_demotion) {
                return mlir::arith::TruncFOp::create(builder, location,
                                                     target_type, value);
            } else {
                return nullptr;
            }
        } else {
            return mlir::arith::ExtFOp::create(builder, location, target_type,
                                               value);
        }
    }

    // Index type conversions
    if (source_type.isIndex() && target_type.isInteger()) {
        auto result = mlir::arith::IndexCastOp::create(builder, location,
                                                       target_type, value);
        SetTypeInfo(result, TypeInfo{target_unsigned.value_or(false)});
        return result;
    }
    if (source_type.isInteger() && target_type.isIndex()) {
        return mlir::arith::IndexCastOp::create(builder, location, target_type,
                                                value);
    }

    // Unsupported conversion - return nullptr to indicate failure
    return nullptr;
}

namespace {
std::pair<mlir::Value, mlir::Value>
ConvertIntegerOperands(mlir::Value lhs, mlir::Value rhs,
                       mlir::OpBuilder &builder, mlir::Location location,
                       bool zeroExtend) {
    auto lhsType = lhs.getType();
    auto rhsType = rhs.getType();
    auto lhsUnsigned = GetTypeInfo(lhs).is_unsigned_integer;
    auto rhsUnsigned = GetTypeInfo(rhs).is_unsigned_integer;
    bool commonUnsigned =
        lhsUnsigned.value_or(false) || rhsUnsigned.value_or(false);
    mlir::Type commonType;
    if (lhsType.isIndex()) {
        commonType = rhsType;
    } else if (rhsType.isIndex()) {
        commonType = lhsType;
    } else {
        commonType =
            lhsType.getIntOrFloatBitWidth() >= rhsType.getIntOrFloatBitWidth()
                ? lhsType
                : rhsType;
    }
    auto convert = [&](mlir::Value value) -> mlir::Value {
        if (zeroExtend && value.getType().isInteger() &&
            commonType.isInteger() &&
            value.getType().getIntOrFloatBitWidth() <
                commonType.getIntOrFloatBitWidth()) {
            auto result = mlir::arith::ExtUIOp::create(builder, location,
                                                       commonType, value);
            SetTypeInfo(result, TypeInfo{commonUnsigned});
            return result;
        }
        return CreateTypeConversion(
            value, value.getType(), commonType, location, builder, false,
            GetTypeInfo(value).is_unsigned_integer, commonUnsigned);
    };
    // Both values must have the complete common type, even if only one widens.
    return {convert(lhs), convert(rhs)};
}
} // namespace

std::pair<mlir::Value, mlir::Value>
ConvertTypesForArithmetic(mlir::Value lhs, mlir::Value rhs,
                          mlir::OpBuilder &builder, mlir::Location location) {
    auto lhsType = lhs.getType();
    auto rhsType = rhs.getType();
    if (lhsType.isIntOrIndex() && rhsType.isIntOrIndex()) {
        return ConvertIntegerOperands(lhs, rhs, builder, location, false);
    }
    if (lhsType == rhsType) {
        return {lhs, rhs};
    }
    if (mlir::isa<mlir::FloatType>(lhsType) &&
        mlir::isa<mlir::FloatType>(rhsType)) {
        if (lhsType.getIntOrFloatBitWidth() < rhsType.getIntOrFloatBitWidth()) {
            return {
                mlir::arith::ExtFOp::create(builder, location, rhsType, lhs),
                rhs};
        }
        if (rhsType.getIntOrFloatBitWidth() < lhsType.getIntOrFloatBitWidth()) {
            return {lhs, mlir::arith::ExtFOp::create(builder, location, lhsType,
                                                     rhs)};
        }
        // Different floating-point formats of the same width (e.g. f16 and
        // bf16) have no implicit common type.
        return {nullptr, nullptr};
    }
    if (lhsType.isInteger() && mlir::isa<mlir::FloatType>(rhsType)) {
        return {CreateTypeConversion(lhs, lhsType, rhsType, location, builder),
                rhs};
    }
    if (rhsType.isInteger() && mlir::isa<mlir::FloatType>(lhsType)) {
        return {lhs,
                CreateTypeConversion(rhs, rhsType, lhsType, location, builder)};
    }
    return {nullptr, nullptr};
}

std::pair<mlir::Value, mlir::Value>
ConvertTypesForBitwise(mlir::Value lhs, mlir::Value rhs,
                       mlir::OpBuilder &builder, mlir::Location location) {
    if (!lhs.getType().isIntOrIndex() || !rhs.getType().isIntOrIndex()) {
        return {nullptr, nullptr};
    }
    // Preserve the existing bit-pattern (zero-extension) policy.
    return ConvertIntegerOperands(lhs, rhs, builder, location, true);
}

} // namespace causalflow::avelang::ir
