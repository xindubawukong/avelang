#pragma once

#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Types.h>
#include <mlir/IR/Value.h>

#include <functional>
#include <map>
#include <optional>
#include <string>

#include "AST/ast_nodes_expr.h"
#include "symbol_scope.h"

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>

namespace causalflow::avelang::ir {

class IRContext;
class NamedModuleRegistry;
class ExprGenerator;

// Represent builtin named modules.
class NamedModule {
  public:
    // Use SymbolScope types
    using TypeFactoryFunction = SymbolScope::TypeFactoryFunction;
    using Function = SymbolScope::Function;

    explicit NamedModule(const std::string &name);
    virtual ~NamedModule() = default;

    const std::string &GetName() const { return name_; }

    // Virtual initialize method for subclasses
    virtual void Initialize() {}
    virtual void DeclareModules(mlir::ModuleOp module) {}

    // Unified symbol lookup
    std::optional<SymbolScope::Symbol>
    LookupSymbol(const std::string &name) const;

  protected:
    friend class NamedModuleRegistry;

    const std::string name_;
    SymbolScope symbols_;

    // Helper methods for adding symbols
    void AddSymbol(const std::string &name, mlir::Value value);
    void AddType(const std::string &name, mlir::Type type, TypeInfo info = {});
    void AddModule(const std::string &name, NamedModule *module);
    void AddTypeFactory(const std::string &name, TypeFactoryFunction factory);
    void AddFunction(const std::string &name,
                     Function::Implementation implementation,
                     Function::Checker checker = nullptr);
};

// Currently it only supports the avelang module
class NamedModuleRegistry {
  public:
    explicit NamedModuleRegistry(IRContext *ir_context);
    void Initialize();
    void DeclareModules(mlir::ModuleOp module);

    NamedModule *GetModule(const std::string &name);

  private:
    IRContext *ir_context_;
    std::map<std::string, std::unique_ptr<NamedModule>> builtin_modules_;
};

} // namespace causalflow::avelang::ir
