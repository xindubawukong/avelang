#include "constant_folder.h"
#include "ir_context.h"
#include "type_promotion.h"
#include "type_system.h"

#include <gtest/gtest.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Verifier.h>

namespace causalflow::avelang::ir {
namespace {

class IntegerSignednessTest : public ::testing::Test {
  protected:
    std::unique_ptr<IRContext> context = IRContext::Create();
    mlir::OpBuilder builder{context->GetMLIRContext()};
    mlir::Location loc = builder.getUnknownLoc();
    mlir::OwningOpRef<mlir::ModuleOp> module{mlir::ModuleOp::create(loc)};

    void SetUp() override {
        builder.setInsertionPointToStart(module->getBody());
    }
    void TearDown() override {
        EXPECT_TRUE(mlir::succeeded(mlir::verify(*module)));
    }

    mlir::Value Integer(unsigned width, bool isUnsigned) {
        auto value =
            mlir::arith::ConstantIntOp::create(builder, loc, -1, width);
        SetTypeInfo(value, TypeInfo{isUnsigned});
        return value;
    }
};

TEST_F(IntegerSignednessTest, ConversionSeparatesSourceAndTargetSignedness) {
    for (unsigned sourceWidth : {8, 16, 32, 64}) {
        for (unsigned targetWidth : {8, 16, 32, 64}) {
            for (bool sourceUnsigned : {false, true}) {
                for (bool targetUnsigned : {false, true}) {
                    SCOPED_TRACE(::testing::Message()
                                 << sourceWidth << '/' << sourceUnsigned
                                 << " -> " << targetWidth << '/'
                                 << targetUnsigned);
                    auto value = Integer(sourceWidth, sourceUnsigned);
                    auto result = CreateTypeConversion(
                        value, value.getType(),
                        builder.getIntegerType(targetWidth), loc, builder, true,
                        sourceUnsigned, targetUnsigned);
                    ASSERT_TRUE(result);
                    EXPECT_EQ(result.getType(),
                              builder.getIntegerType(targetWidth));
                    EXPECT_EQ(GetTypeInfo(result).is_unsigned_integer,
                              targetUnsigned);
                    EXPECT_EQ(GetTypeInfo(value).is_unsigned_integer,
                              sourceUnsigned);
                    if (sourceWidth < targetWidth) {
                        EXPECT_EQ(mlir::isa<mlir::arith::ExtUIOp>(
                                      result.getDefiningOp()),
                                  sourceUnsigned);
                        EXPECT_EQ(mlir::isa<mlir::arith::ExtSIOp>(
                                      result.getDefiningOp()),
                                  !sourceUnsigned);
                    }
                    if (sourceWidth > targetWidth) {
                        EXPECT_FALSE(CreateTypeConversion(
                            value, value.getType(),
                            builder.getIntegerType(targetWidth), loc, builder,
                            false, sourceUnsigned, targetUnsigned));
                    }
                }
            }
        }
    }
}

TEST_F(IntegerSignednessTest, CommonIntegerTypeIsIndependentOfOperandOrder) {
    for (unsigned lhsWidth : {8, 16, 32, 64}) {
        for (unsigned rhsWidth : {8, 16, 32, 64}) {
            for (bool lhsUnsigned : {false, true}) {
                for (bool rhsUnsigned : {false, true}) {
                    auto lhs = Integer(lhsWidth, lhsUnsigned);
                    auto rhs = Integer(rhsWidth, rhsUnsigned);
                    for (bool swapped : {false, true}) {
                        auto [a, b] = ConvertTypesForArithmetic(
                            swapped ? rhs : lhs, swapped ? lhs : rhs, builder,
                            loc);
                        ASSERT_TRUE(a && b);
                        EXPECT_EQ(a.getType(), builder.getIntegerType(std::max(
                                                   lhsWidth, rhsWidth)));
                        EXPECT_EQ(a.getType(), b.getType());
                        EXPECT_EQ(GetTypeInfo(a).is_unsigned_integer,
                                  lhsUnsigned || rhsUnsigned);
                        EXPECT_EQ(GetTypeInfo(b).is_unsigned_integer,
                                  lhsUnsigned || rhsUnsigned);
                    }
                    EXPECT_EQ(GetTypeInfo(lhs).is_unsigned_integer,
                              lhsUnsigned);
                    EXPECT_EQ(GetTypeInfo(rhs).is_unsigned_integer,
                              rhsUnsigned);
                }
            }
        }
    }
}

TEST_F(IntegerSignednessTest, BooleanConversionsZeroExtend) {
    auto boolean = Integer(1, false);
    auto widened = CreateTypeConversion(boolean, boolean.getType(),
                                        builder.getI32Type(), loc, builder);
    EXPECT_TRUE(widened.getDefiningOp<mlir::arith::ExtUIOp>());
    auto floating = CreateTypeConversion(boolean, boolean.getType(),
                                         builder.getF32Type(), loc, builder);
    EXPECT_TRUE(floating.getDefiningOp<mlir::arith::UIToFPOp>());
}

TEST_F(IntegerSignednessTest, BitwiseWideningKeepsZeroExtensionPolicy) {
    auto [lhs, rhs] = ConvertTypesForBitwise(Integer(32, false),
                                             Integer(64, false), builder, loc);
    EXPECT_TRUE(lhs.getDefiningOp<mlir::arith::ExtUIOp>());
    EXPECT_EQ(GetTypeInfo(lhs).is_unsigned_integer, false);
    EXPECT_EQ(GetTypeInfo(rhs).is_unsigned_integer, false);
}

TEST_F(IntegerSignednessTest, FoldingUsesInstructionWidthsAndPredicates) {
    for (unsigned width : {8, 16, 32, 64}) {
        auto minusOne = Integer(width, false);
        auto one = mlir::arith::ConstantIntOp::create(builder, loc, 1, width);
        auto sum = mlir::arith::AddIOp::create(builder, loc, minusOne, one);
        EXPECT_EQ(ConstantFolder::FoldIntValue(sum), 0);
        auto logical =
            mlir::arith::ShRUIOp::create(builder, loc, minusOne, one);
        EXPECT_EQ(ConstantFolder::FoldIntValue(logical),
                  static_cast<int64_t>(
                      llvm::APInt::getSignedMaxValue(width).getZExtValue()));
        auto arithmetic =
            mlir::arith::ShRSIOp::create(builder, loc, minusOne, one);
        EXPECT_EQ(ConstantFolder::FoldIntValue(arithmetic), -1);
        auto less = mlir::arith::CmpIOp::create(
            builder, loc, mlir::arith::CmpIPredicate::ult, minusOne, one);
        EXPECT_EQ(ConstantFolder::FoldBoolValue(less), false);
        auto signedLess = mlir::arith::CmpIOp::create(
            builder, loc, mlir::arith::CmpIPredicate::slt, minusOne, one);
        EXPECT_EQ(ConstantFolder::FoldBoolValue(signedLess), true);
        auto retagged = CreateTypeConversion(minusOne, minusOne.getType(),
                                             minusOne.getType(), loc, builder,
                                             false, false, true);
        EXPECT_EQ(ConstantFolder::FoldIntValue(retagged),
                  static_cast<int64_t>(
                      llvm::APInt::getAllOnes(width).getZExtValue()));
    }
    auto minusOne = Integer(32, false);
    auto unsignedWide =
        CreateTypeConversion(minusOne, minusOne.getType(), builder.getI64Type(),
                             loc, builder, true, true, false);
    EXPECT_EQ(ConstantFolder::FoldIntValue(unsignedWide), 4294967295);
    auto signedWide =
        CreateTypeConversion(minusOne, minusOne.getType(), builder.getI64Type(),
                             loc, builder, true, false, false);
    EXPECT_EQ(ConstantFolder::FoldIntValue(signedWide), -1);
    auto truncated = CreateTypeConversion(unsignedWide, unsignedWide.getType(),
                                          builder.getI8Type(), loc, builder,
                                          true, false, false);
    EXPECT_EQ(ConstantFolder::FoldIntValue(truncated), -1);
    EXPECT_EQ(truncated.getDefiningOp()->getOperand(0), unsignedWide);
}

TEST_F(IntegerSignednessTest, FoldingDoesNotEvaluatePoison) {
    auto one = mlir::arith::ConstantIntOp::create(builder, loc, 1, 32);
    auto zero = mlir::arith::ConstantIntOp::create(builder, loc, 0, 32);
    auto width = mlir::arith::ConstantIntOp::create(builder, loc, 32, 32);
    auto min = mlir::arith::ConstantIntOp::create(builder, loc, INT32_MIN, 32);
    auto minusOne = Integer(32, false);
    EXPECT_FALSE(ConstantFolder::FoldIntValue(
        mlir::arith::DivUIOp::create(builder, loc, one, zero)));
    EXPECT_FALSE(ConstantFolder::FoldIntValue(
        mlir::arith::DivSIOp::create(builder, loc, min, minusOne)));
    EXPECT_FALSE(ConstantFolder::FoldIntValue(
        mlir::arith::ShLIOp::create(builder, loc, one, width)));
}

} // namespace
} // namespace causalflow::avelang::ir
