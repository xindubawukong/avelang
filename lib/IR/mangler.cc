#include "mangler.h"

#include "AST/ast_nodes_expr.h"
#include "AST/ast_nodes_stmt.h"

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wambiguous-reversed-operator"
#include <mlir/Dialect/GPU/IR/GPUDialect.h>
#include <mlir/IR/BuiltinAttributes.h>
#pragma clang diagnostic pop

#include <cctype>
#include <optional>

#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringExtras.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/SHA256.h>
#include <llvm/Support/raw_ostream.h>

namespace causalflow::avelang::ir {

namespace {

std::string SanitizeManglePart(llvm::StringRef value) {
    std::string result;
    result.reserve(value.size());
    for (char c : value) {
        unsigned char uc = static_cast<unsigned char>(c);
        if (std::isalnum(uc)) {
            result.push_back(c);
        } else if (c == '\'') {
            result.append("_sq_");
        } else if (c == '-') {
            result.push_back('m');
        } else {
            result.push_back('_');
        }
    }
    return result;
}

std::string MangleAddressSpace(mlir::Attribute memorySpace) {
    if (!memorySpace) {
        return "default";
    }
    if (auto gpuSpace =
            mlir::dyn_cast<mlir::gpu::AddressSpaceAttr>(memorySpace)) {
        switch (gpuSpace.getValue()) {
        case mlir::gpu::AddressSpace::Global:
            return "global";
        case mlir::gpu::AddressSpace::Workgroup:
            return "workgroup";
        case mlir::gpu::AddressSpace::Private:
            return "private";
        default:
            break;
        }
        return "as" + std::to_string(static_cast<int>(gpuSpace.getValue()));
    }
    if (auto intSpace = mlir::dyn_cast<mlir::IntegerAttr>(memorySpace)) {
        return "as" + std::to_string(intSpace.getInt());
    }
    return "unknown";
}

std::string MangleConstexprValueTag(ConstexprValue value) {
    std::string storage;
    llvm::raw_string_ostream os(storage);
    value.value.print(os);
    os << ";unsigned=" << value.type_info.is_unsigned_integer.value_or(false);
    auto digest = llvm::SHA256::hash(llvm::arrayRefFromStringRef(storage));
    return llvm::toHex(llvm::ArrayRef(digest), /*LowerCase=*/true);
}

template <typename T>
std::optional<T>
FindBinding(llvm::ArrayRef<std::pair<std::string, T>> bindings,
            llvm::StringRef name) {
    for (const auto &[bindingName, value] : bindings) {
        if (bindingName == name) {
            return value;
        }
    }
    return std::nullopt;
}

bool IsConstexprArg(ast::Arg *arg) {
    auto *attrExpr =
        llvm::dyn_cast_or_null<ast::AttributeExpr>(arg->GetAnnotation());
    return attrExpr && attrExpr->GetAttr() == "constexpr";
}

std::string BuildMangledName(llvm::ArrayRef<std::string> scope,
                             llvm::StringRef name,
                             llvm::ArrayRef<std::string> addressSpaceTags,
                             llvm::ArrayRef<std::string> constexprTags) {
    if (name.empty()) {
        return {};
    }

    size_t totalSize = name.size();
    for (const auto &part : scope) {
        totalSize += part.size() + 1;
    }
    if (!addressSpaceTags.empty()) {
        totalSize += 4; // "__as"
        for (const auto &tag : addressSpaceTags) {
            totalSize += tag.size() + 1;
        }
    }
    if (!constexprTags.empty()) {
        totalSize += 4; // "__ce"
        for (const auto &tag : constexprTags) {
            totalSize += tag.size() + 1;
        }
    }

    std::string mangled;
    mangled.reserve(totalSize);
    for (const auto &part : scope) {
        if (!mangled.empty()) {
            mangled.push_back('_');
        }
        mangled.append(part);
    }
    if (!mangled.empty()) {
        mangled.push_back('_');
    }
    mangled.append(name.data(), name.size());
    if (!addressSpaceTags.empty()) {
        mangled.append("__as");
        for (const auto &tag : addressSpaceTags) {
            mangled.push_back('_');
            mangled.append(tag);
        }
    }
    if (!constexprTags.empty()) {
        mangled.append("__ce");
        for (const auto &tag : constexprTags) {
            mangled.push_back('_');
            mangled.append(tag);
        }
    }
    return mangled;
}

} // namespace

std::string MangleFunctionName(
    ast::FunctionDef *func, llvm::ArrayRef<std::string> scope,
    llvm::ArrayRef<std::pair<std::string, mlir::Attribute>> addressSpaces,
    llvm::ArrayRef<std::pair<std::string, ConstexprValue>> constexprValues) {
    if (!func) {
        return {};
    }

    llvm::SmallVector<std::string, 4> addressSpaceTags;
    llvm::SmallVector<std::string, 4> constexprTags;
    if (auto *args = func->GetArguments()) {
        for (auto *arg : args->GetArgs()) {
            if (!arg) {
                continue;
            }
            const auto &argName = arg->GetArgName();
            if (IsConstexprArg(arg)) {
                if (auto value = FindBinding(constexprValues, argName)) {
                    constexprTags.push_back(SanitizeManglePart(argName) + "_" +
                                            MangleConstexprValueTag(*value));
                }
                continue;
            }

            auto addressSpace = FindBinding(addressSpaces, argName);
            if (addressSpace) {
                addressSpaceTags.push_back(MangleAddressSpace(*addressSpace));
            }
        }
    }

    return BuildMangledName(scope, func->GetName(), addressSpaceTags,
                            constexprTags);
}

} // namespace causalflow::avelang::ir
