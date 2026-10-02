#include "symbol_scope.h"
#include "named_module.h"

#include <utility>

namespace causalflow::avelang::ir {

SymbolScope::Symbol::Symbol() : kind(kValue), value() {}

SymbolScope::Symbol::Symbol(NamedModule *m) : kind(kModule), module(m) {}

SymbolScope::Symbol::Symbol(TypeFactoryFunction tf)
    : kind(kTypeFactory), type_factory(tf) {}

SymbolScope::Symbol::Symbol(Function ff)
    : kind(kFunction), function(std::move(ff)) {}

SymbolScope::Symbol::Symbol(mlir::Type t) : kind(kType), type(t) {}

SymbolScope::Symbol::Symbol(mlir::Value v) : kind(kValue), value(v) {}
SymbolScope::Symbol::Symbol(ConstexprValue v)
    : kind(kConstexpr), constexpr_value(v) {}

SymbolScope::Symbol::~Symbol() {
    switch (kind) {
    case kTypeFactory:
        type_factory.~TypeFactoryFunction();
        break;
    case kFunction:
        function.~Function();
        break;
    case kConstexpr:
        constexpr_value.~ConstexprValue();
        break;
    case kModule:
    case kType:
    case kValue:
        // These types don't need explicit destruction
        break;
    }
}

SymbolScope::Symbol::Symbol(const Symbol &other) : kind(other.kind) {
    switch (kind) {
    case kModule:
        module = other.module;
        break;
    case kTypeFactory:
        new (&type_factory) TypeFactoryFunction(other.type_factory);
        break;
    case kFunction:
        new (&function) Function(other.function);
        break;
    case kType:
        type = other.type;
        break;
    case kValue:
        value = other.value;
        break;
    case kConstexpr:
        new (&constexpr_value) ConstexprValue(other.constexpr_value);
        break;
    }
}

SymbolScope::Symbol &SymbolScope::Symbol::operator=(const Symbol &other) {
    if (this != &other) {
        // Destroy current content
        this->~Symbol();

        // Copy construct new content
        kind = other.kind;
        switch (kind) {
        case kModule:
            module = other.module;
            break;
        case kTypeFactory:
            new (&type_factory) TypeFactoryFunction(other.type_factory);
            break;
        case kFunction:
            new (&function) Function(other.function);
            break;
        case kType:
            type = other.type;
            break;
        case kValue:
            value = other.value;
            break;
        case kConstexpr:
            new (&constexpr_value) ConstexprValue(other.constexpr_value);
            break;
        }
    }
    return *this;
}

bool SymbolScope::Symbol::isa(SymbolKind k) const { return this->kind == k; }

void SymbolScope::AddSymbol(const std::string &name, const Symbol &symbol) {
    symbols_[name] = symbol;
}

std::optional<SymbolScope::Symbol>
SymbolScope::LookupSymbol(const std::string &name) const {
    auto it = symbols_.find(name);
    if (it != symbols_.end()) {
        return it->second;
    }
    return std::nullopt;
}

void SymbolScope::AddValue(const std::string &name, mlir::Value value) {
    symbols_[name] = Symbol(value);
}

void SymbolScope::AddType(const std::string &name, mlir::Type type) {
    symbols_[name] = Symbol(type);
}

void SymbolScope::AddConstexpr(const std::string &name, ConstexprValue value) {
    symbols_[name] = Symbol(value);
}

void SymbolScope::AddModule(const std::string &name, NamedModule *module) {
    symbols_[name] = Symbol(module);
}

void SymbolScope::AddTypeFactory(const std::string &name,
                                 TypeFactoryFunction factory) {
    symbols_[name] = Symbol(factory);
}

void SymbolScope::AddFunction(const std::string &name,
                              Function::Implementation implementation,
                              Function::Checker checker) {
    symbols_[name] = Symbol(
        Function(std::move(implementation), std::move(checker), {}, name));
}

void SymbolScope::AddFunction(const std::string &name, Function function) {
    symbols_[name] = Symbol(std::move(function));
}

mlir::Value SymbolScope::LookupValue(const std::string &name) const {
    auto it = symbols_.find(name);
    if (it != symbols_.end() && it->second.isa(kValue)) {
        return it->second.value;
    }
    return mlir::Value();
}

mlir::Type SymbolScope::LookupType(const std::string &name) const {
    auto it = symbols_.find(name);
    if (it != symbols_.end() && it->second.isa(kType)) {
        return it->second.type;
    }
    return mlir::Type();
}

NamedModule *SymbolScope::LookupModule(const std::string &name) const {
    auto it = symbols_.find(name);
    if (it != symbols_.end() && it->second.isa(kModule)) {
        return it->second.module;
    }
    return nullptr;
}

SymbolScope::TypeFactoryFunction
SymbolScope::LookupTypeFactory(const std::string &name) const {
    auto it = symbols_.find(name);
    if (it != symbols_.end() && it->second.isa(kTypeFactory)) {
        return it->second.type_factory;
    }
    return nullptr;
}

SymbolScope::Function
SymbolScope::LookupFunction(const std::string &name) const {
    auto it = symbols_.find(name);
    if (it != symbols_.end() && it->second.isa(kFunction)) {
        return it->second.function;
    }
    return Function();
}

} // namespace causalflow::avelang::ir
