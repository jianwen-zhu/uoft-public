# YTL Language Reference Manual 2.0

Jianwen Zhu

Version 2.0

## Status and normative language

This document is the normative definition of YTL 2.0. The accompanying
reference parser, elaborator, verifier, and interpreter implement this
specification. Earlier 0.1 artifacts remain governed by the
[YTL 0.1 manual](ytl-lrm.md) and the [TTL 0.1 manual](ttl-lrm.md).

The key words **MUST**, **MUST NOT**, **REQUIRED**, **SHALL**, **SHALL NOT**,
**SHOULD**, **SHOULD NOT**, **MAY**, and **OPTIONAL** are to be interpreted as
described by RFC 2119 and RFC 8174 when, and only when, they appear in bold.

Examples are illustrative unless a requirement explicitly makes them
normative. Project plans, implementation notes, and design rationale do not
extend this specification.

## 1. Introduction

### 1.1 Purpose and boundary

YTL, the **Yang Tensor Language**, is a first-order functional language for
computations over multidimensional arrays, called **tensors**. A tensor maps
coordinates to element values; each independent coordinate dimension is
called an **axis**.
“First-order” means functions may be called but are not passed or returned as
values. “Functional” means expressions produce immutable values rather than
updating them in place.

YTL is intended for mathematical tensor programs: machine-learning models,
structured tensor algorithms, and interconnection networks. Its compile-time
composition mechanisms are also intended to support hardware construction in
future versions. A reference implementation evaluates YTL and TTL directly;
independent compilers may consume TTL for CPUs, GPUs, tensor accelerators, or
hardware-description flows.

YTL exposes the values a computation produces and the mathematical
relationships among their axes. It does not prescribe how those values are
stored or scheduled. Physical layout, tiling, kernel boundaries, memory
placement, synchronization, machine-code generation, checkpoint formats,
tokenization, and host orchestration are outside the language. The intended
implementation targets motivate this separation between inherent computation
and implementation detail, but none is part of the language semantics.

An ordinary programming language typically describes arrays through ordered
integer positions and describes their computation through loops, indexing,
mutation, and control flow. A tensor library improves the notation but usually
continues to identify a dimension as “axis 1” or “the last axis.” Correctness
then depends on knowledge that is absent from the program: which position is
the token dimension, which equal-sized dimension is the head dimension, and
which dimensions a helper silently preserves.

YTL instead makes coordinate meaning a first-class citizen of the language. A
tensor type names each independent coordinate dimension, and tensor
expressions align those names. A misspelled, swapped, or semantically unrelated
dimension is therefore a type error even when every integer extent happens to
agree. Expressions derive their result types from their operands and
operations; a left-hand side does not supply a missing shape.

Before execution, YTL source is *elaborated*: compile-time computation is
evaluated, helper functions are specialized and inlined, shorthand is
expanded, and structured source is reduced to a finite graph of explicit
tensor primitives. The resulting language is TTL, the **Tiny Tensor Language**.
This separation lets YTL remain pleasant for authors while giving interpreters
and downstream compilers a finite, explicit, implementation-neutral input.

### 1.2 Vocabulary

The following terms are used throughout this manual:

- An **element type** describes one tensor element, for example `f32`.
- An **axis** names one independent coordinate dimension. Its identity and
  its extent are distinct properties.
- An **axis domain** is the set of valid coordinates for one axis occurrence.
  A static domain is described by an extent; a dynamic domain is supplied at
  runtime for each coordinate assignment over its declared dependencies.
- A **coordinate** is a finite mapping that chooses one position for every
  named axis of a tensor.
- A **tensor** maps every valid named coordinate to one element.
- The **rank** of a tensor is its number of axes. A **scalar** is a rank-zero
  tensor.
- A **value** is an immutable scalar or tensor. A function may also return a
  fixed list of values, called a **multiple result**.
- A **tensor type** consists of an element type and a finite mapping from named
  axis identities to axis domains. The mapping has no inter-axis order.
- A **static parameter** is a name declared between `[` and `]` in a function
  header and resolved during elaboration. A **value parameter** is declared
  between `(` and `)` and receives a runtime value.
- A **module** is one YTL source unit. A **concrete axis** is one module-wide
  identity. A **generic axis** is an axis placeholder declared between `[` and
  `]` in a function header and specialized with an axis identity at a call. It
  is not a runtime value parameter; runtime value parameters occur between
  `(` and `)`.
- A **static value** is known and evaluated during elaboration rather than
  tensor execution.
- A **primitive** is a compiler-owned tensor operation written with an
  `@`-name, such as `@einsum`.
- A **target environment** supplies scalar types and scalar operations. A
  declared **target function** is one such compiler-owned scalar operation;
  “target” does not imply a code-generation backend in this specification.
- **Elaboration** is the compile-time reduction of YTL source to TTL. It must
  terminate and produce a finite TTL module; runtime tensor values cannot
  determine the structure of that module.
- **TTL**, the Tiny Tensor Language, is the flat, fully typed, primitive-only
  language produced by elaboration.
- An axis **scope** is a `let [axes] ... in ...` expression and its lexical
  region.
- A scope's **prefix** is the named coordinate environment established by its
  header together with the environments of its enclosing scopes.
- A **fibre** is the remaining tensor seen after those prefix coordinates are
  fixed.
- **Lifting** applies a fibre computation at every coordinate of a prefix and
  restores that prefix to the result type.

### 1.3 Defining ideas

YTL combines five ideas:

1. **Semantic tensor types.** Axis identity participates in type checking;
   equal extents do not imply alignment.
2. **Einstein relationships.** Mapping, broadcasting, contraction, gather,
   scatter, and coordinate transformation are stated with named axes rather
   than positional shape arithmetic.
3. **A minimal primitive core.** Rich source notation elaborates to a small,
   explicit, primitive-only language rather than accumulating independent
   runtime mechanisms.
4. **Finite compile-time composition.** Static parameters, conditionals, and
   structural recursion express families only when elaboration can reduce each
   admitted program to a finite TTL graph.
5. **Scoped fibres.** Repeated prefix axes may be factored out while the local
   expression describes only the mathematical fibre. Flattening restores an
   equivalent ordinary tensor graph.

The first four ideas form the YTL 1.0 profile. Scoped fibres and APL-style
map-reduce notation are the two additions made by YTL 2.0.

### 1.4 A complete program

```ttl
import ttl.lang;

axis batch, token, hidden, vocabulary;

global embedding: f16[vocabulary hidden];

function project[h, o](
    x: f16[h],
    weight: f16[o h]
) -> f16[o] :=
    @einsum[h, o h -> o](x, weight);

export function logits(
    x: f16[batch token hidden]
) -> f16[batch token vocabulary] :=
    let [batch token] in
        project[hidden, vocabulary](x, embedding);
```

The module declares four concrete axes and one persistent tensor value.
`project` is generic over its two axis parameters. In `logits`, the scoped
expression fixes one `batch, token` coordinate at a time. Its local call sees
an `f16[hidden]` fibre of `x`, returns an `f16[vocabulary]` fibre, and lifting
produces `f16[batch token vocabulary]`.

## 2. Language notation

### 2.1 Lexical structure

Source text is Unicode, but every token defined by YTL 2.0 is ASCII.
Whitespace and comments may occur between tokens and are discarded. They
cannot occur within a token. Line comments begin with `//`. Block comments are
delimited by `/*` and `*/` and do not nest.

```ebnf
id          ::= /[A-Za-z_][A-Za-z0-9_]*/ ;
integer     ::= /0|[1-9][0-9]*/ ;
float       ::= /([0-9]+\.[0-9]*|\.[0-9]+)([eE][+-]?[0-9]+)?/
              | /[0-9]+[eE][+-]?[0-9]+/ ;
at-id       ::= /@[A-Za-z_][A-Za-z0-9_]*/ ;
perc-id     ::= /%(0|[1-9][0-9]*)/ ;
dollar-id   ::= /\$[A-Za-z_][A-Za-z0-9_]*/ ;
```

`at-id`, `perc-id`, and `dollar-id` are each one token; whitespace cannot
separate the prefix from its body. Leading zeroes are forbidden in a
multi-digit integer or `perc-id` number. Quoted words in the syntactic grammar
are reserved tokens and are not `id` tokens. Quoted punctuation in that
grammar also denotes tokens. A negative numeric literal consists of `-`
followed by an unsigned literal.

### 2.2 YTL 1.0 common grammar

`X?`, `X*`, and `X+` mean optional, zero-or-more, and one-or-more.

The following is the complete grammar shared by YTL 1.0 and YTL 2.0:

```ebnf
program   ::= import* axis-decl* global* function* EOF ;
import    ::= "import" id ("." id)* ";" ;
axis-decl ::= "axis"
              axis-definition ("," axis-definition)* ";" ;
axis-definition ::= static-axis-definition | dynamic-axis-definition ;
static-axis-definition ::= id (":=" expr)? ;
dynamic-axis-definition ::= dollar-id ("over" axis+)? ;
global    ::= "global" id ":" type ";" ;

function  ::= marker* "function" name static-params?
              "(" (param ("," param)*)? ")" ("->" results)?
              ("identity" "=" expr)* (":=" expr)? ";" ;
marker    ::= "export" | "reducer" | "target" ;
static-params ::= "[" id ("," id)* "]" ;
param      ::= id (":" type)? ;
results    ::= type | "(" type ("," type)+ ")" ;

type       ::= id ("[" dimension* "]")? ;
dimension  ::= static-dimension | dynamic-dimension ;
static-dimension ::= id (":=" expr)? ;
dynamic-dimension ::= dollar-id ;

binder     ::= id | perc-id
             | "(" (id | perc-id) ("," (id | perc-id))+ ")" ;
binding    ::= binder (":" results)? "=" expr ;

expr       ::= let-expr | if-expr | infix-expr ;
let-expr   ::= "let" binding (";" binding)* "in" expr ;
if-expr    ::= "if" expr "then" expr "else" expr ;
infix-expr ::= prefix (binary-op prefix)* ;
prefix     ::= ("-" | "!") prefix | primary ;
primary    ::= atom selector* ;
atom       ::= literal | id | perc-id
             | id static-args? "(" (expr ("," expr)*)? ")"
             | primitive
             | "(" expr ")"
             | "(" expr ("," expr)+ ")" ;
static-args ::= "[" static-arg ("," static-arg)* "]" ;
static-arg  ::= expr | axis ;
selector   ::= "[" selection ("," selection)* "]" ;
selection  ::= axis "=" expr (":" expr)? ;

primitive  ::= at-id ("[" axis-spec "]")?
              "(" (expr ("," expr)*)? ")"
              (("map" | "reduce") "=" name)* ;
axis-spec  ::= axis-pattern ("," axis-pattern)* "->" axis-pattern ;
axis-pattern ::= (dimension | axis-product)* ;
axis-product ::= axis ":=" "[" axis ("*" axis)+ "]" ;

name       ::= id | binary-op | "!" | at-id ;
binary-op  ::= "||" | "&&" | "==" | "!=" | "<" | "<=" | ">" | ">="
             | "+" | "-" | "*" | "/" | "**" ;
axis       ::= id | dollar-id ;
literal    ::= integer | float | "inf" | "true" | "false" ;
```

Thus an axis pattern is a sequence of only three item forms: an axis `a`, an
axis whose domain is defined by an extent `a := expr`, or a one-level axis
product `p := [a * b * ...]`. The `:=` token defines an axis domain, either by
an extent or by a product; square brackets distinguish a structural product
from an extent expression, and `*` separates the product's ordered factors. A
product has at least two factors, and each factor is only an axis, so the
notation defines neither aliasing nor a recursive or nested axis structure.

YTL distinguishes definition from binding. `:=` defines the body of a declared
function or the domain of an axis; `=` binds a local value, selector coordinate
or range, or named clause within an enclosing construct. `==` compares values.
These roles do not overlap.

A literal is an ordinary YTL atom, and every primitive argument is an
expression. Authors may therefore write literals and computed expressions
directly:

```ttl
function scale(x: f32[row]) -> f32[row] := x * 2.0;

function scale_explicit(x: f32[row]) -> f32[row] :=
    @einsum[row, -> row](x, 2.0) map=*;
```

Elaboration names such immediate values and intermediate computations when it
produces flat TTL. YTL source does not require authors to introduce those
administrative bindings.

Application source **MUST NOT** define a `perc-id`. A multiple binder and a
multiple result contain at least two members by construction.

### 2.3 YTL 2.0 grammar additions

YTL 2.0 changes exactly two expression productions. It adds a scoped `let`:

```ebnf
expr       ::= let-expr | scope-expr | if-expr | infix-expr ;
scope-expr ::= "let" "[" dimension+ "]"
               (binding (";" binding)*)? "in" expr ;
```

It also admits a map-reduce operator in an infix expression:

```ebnf
infix-expr    ::= prefix ((binary-op | map-reduce-op) prefix)* ;
map-reduce-op ::= name "." name "[" axis+ "]" ;
```

Removing `scope-expr` and restricting `infix-expr` to `binary-op` recovers the
complete YTL 1.0 grammar. No other syntax differs between the profiles.

### 2.4 Operator precedence

Binary operators except `**` are left-associative. `**` is
right-associative. Precedence from lowest to highest is:

```text
||
&&
== !=
< <= > >=
+ -
* / and map-reduce operators
**
prefix - !
selectors
```

A map-reduce operator is non-associative. Chained map-reduce expressions
require parentheses.

## 3. Type system

### 3.1 Scalar and tensor types

A scalar type is a dtype without axes:

```ttl
bool
i32
f16
f32
```

Appending named axes produces a tensor type:

```ttl
f32[row column]
f16[batch $token hidden]  // with axis $token over batch
```

Semantically, each axis entry contains both an identity and its current
domain. For a static axis, an extent `N` denotes coordinates `0 .. N-1`:

```ttl
f32[token := 8]
```

The type therefore does not consist of axis names alone. Two occurrences of
`token` may have different local extents after an explicit operation such as a
slice; they do not align until both identity and current domain agree.

| Axis occurrence | Semantic identity | Current domain | When determined |
|---|---|---|---|
| `f32[token]`, with `axis token := 8` | `token` | `0 .. 7` | elaboration |
| `x[token=2:6]` | `token` | `0 .. 3`, reindexed | elaboration |
| `f32[batch $past]`, with `axis $past over batch` | `$past` | `0 .. length_past(batch)-1` | runtime descriptor |

The slice offset `2` determines which values the selector copies; it is not
part of the resulting axis domain. Section 7.3 defines this rule completely.

Axis identity, not written position, determines alignment and ordinary type
compatibility. An ordinary whitespace-separated axis list denotes a finite
set. Consequently, these types are equal:

```ttl
f32[batch token hidden]
f32[hidden batch token]
f32[token hidden batch]
```

Permuting axis entries in a tensor type or ordinary axis pattern does not
change meaning. A tensor type cannot contain the same axis twice after generic
specialization. This rule does not erase the dynamic-axis dependency partial
order defined in Section 3.5. Axis-product factors and operands in a
concatenating `@rearrange` retain their local order.

### 3.2 Concrete axes

A module `axis` declaration introduces a concrete semantic identity:

```ttl
axis batch, token, hidden := 4096, $past over batch;
```

All concrete axes used by a module **MUST** be declared exactly once. Their
declarations are collected before type checking, so order is irrelevant.
Every occurrence of one declared spelling denotes the same identity in that
module.

A declaration of `a := expr` defines the base extent of static axis `a`. A bare
occurrence of `a` uses that base extent unless specialization or an explicit
operation supplies a local domain. A dynamic axis such as `$past` **MUST NOT**
have an extent expression. Its optional `over` clause names its immediate
parent axes. Only a dynamic concrete axis may have such a clause. Declaring
two names never aliases them.

### 3.3 Generic axes

A name declared in a function's static-parameter list becomes a generic axis
when it is used in an axis position:

```ttl
function identity[a](x: f32[a]) -> f32[a] := x;
```

At a call, the corresponding static argument supplies a concrete axis
identity:

```ttl
identity[token](x)
```

Substitution **MUST** be capture-avoiding and injective within one call. Two
distinct generic axes cannot be specialized to one identity. A generic is a
placeholder, not a type alias.

### 3.4 Static-axis resolution

An axis name written without the lexical `$` prefix is a **static axis**. It
has a positive extent resolved during elaboration. An occurrence `a := expr` is a
constraint on the current domain of that occurrence, not a runtime assignment.
All constraints contributing to one domain **MUST** have one consistent
positive solution.

```ttl
axis batch := 2, hidden := 4096, $past over batch;
global cache: f16[batch $past hidden];
```

Here the extents of `batch` and `hidden` are resolved from their declarations.
At execution, the value supplied for `cache` carries the runtime domain of
`$past`.

A static expression may use integer and Boolean literals, static parameters,
resolved static-axis extents, parentheses, `+`, `-`, `*`, `**`, comparisons,
equality, `!`, `&&`, and `||`. Exponents **MUST** be non-negative. Division is
not a static operation in YTL 2.0.

### 3.5 Dynamic axes

A `$`-prefixed axis has a runtime-varying extent. Its declaration may name its
immediate parents with `over`:

```ttl
axis batch, $sentence over batch, $token over $sentence;
```

Let `parents(a)` be the unordered set written after `over`. The strict
transitive closure of `parents` is the semantic dependency partial order, and
`support(a)` is the set of all predecessors of `a`. Parent axes **MUST** be
distinct, **MUST NOT** include `a`, and **MUST** be declared concrete axes. The
graph **MUST** be acyclic. A dynamic axis without `over` is a root and has one
runtime extent for the whole tensor.

The descriptor for `$a` is a logical function `length_a(s)` from every named
coordinate assignment `s` over `support(a)` to a non-negative integer. At that
coordinate assignment, valid `$a` coordinates are
`0 .. length_a(s)-1`. A zero extent is valid.

Fixing prefix coordinates partially applies a dynamic descriptor. Two
occurrences of one dynamic axis align only when they have the same identity,
the same residual support set, and equal current runtime extents at every
residual support coordinate. A primitive may define an explicit
transformation between otherwise different dynamic domains.

### 3.6 Contextual type well-formedness

Let `A` be the axis set of tensor type `T[A]`, let `P` be the axis set fixed by
the effective enclosing scope prefix, and let `support(d)` be the predecessors
of dynamic axis `d`. Fixing `P` partially applies the descriptor of `d`; its
residual support set is `support(d) minus P`. The contextual well-formedness
judgment is:

```text
WF_P(T[A]) iff
    every axis in A is distinct, and
    for every dynamic d in A: support(d) minus P is a subset of A
```

The empty prefix is used outside a scope. A prefix is itself well formed only
when every dynamic axis in it has its support set in that prefix. Every
declared concrete parameter, result, and global type, and every inferred
expression and binding type, **MUST** satisfy the judgment in its lexical
context.

A generic axis has no concrete support set before specialization. Its
declaration is checked symbolically. At each call, substitution supplies a
concrete identity, and the current domain is taken from the matching actual
arguments or an explicit result-producing coordinate relation. The
specialized parameter and result types **MUST** satisfy the judgment in the
caller's context. A fibre call passes the already partially applied descriptor;
the callee does not receive or inspect the caller's prefix coordinates.

Every tensor-producing rule first derives its result axis mapping and then
applies this judgment. Consequently, no selector, reduction, gather,
rearrangement, call, conditional, or lifted scalar operation may remove a
dependency while retaining the dependent dynamic axis. For example:

```ttl
axis batch, $token over batch;

function invalid(x: f32[batch $token]) -> f32[$token] :=
    x[batch=0];
```

This function is invalid because `$token` still requires `batch`. A scope may
instead fix `batch`; within that scope `f32[$token]` is well formed because
`batch` belongs to the prefix, and lifting restores `batch` on scope exit.
For the same reason, `@einsum[batch $token -> $token](x)` is invalid: reducing
`batch` cannot leave its dependent `$token` in the result.

### 3.7 Explicit coordinate transformations

```ttl
@rearrange[hidden := [head * channel] -> head channel](x)
```

If `x` has type `f32[hidden]`, the result has type `f32[head channel]`.
The primitive checks the product relation while evaluating this expression.
The result type contains no evidence that `head` and `channel` came from
`hidden`.

YTL 2.0 has no persistent “derived axis” category. `@rearrange` states and
checks a coordinate relation for that operation only. Its result has ordinary
axis identities and domains; the type does not retain parent/child lineage.
Applying another rearrangement requires stating and checking another complete
relation.

A selector similarly computes a new value and may replace one static axis
domain with a smaller, reindexed domain. The result type retains the axis
identity and its new extent, but not the source offset. Section 7.3 gives the
exact rule.

### 3.8 Type equality and compatibility

Two tensor types are equal exactly when their element types and their mappings
from axis identities to current static extents or dynamic descriptors are
equal. Written order is never a type-equality criterion. Dynamic dependency is
part of the axis declaration and cannot be changed by moving a spelling within
a type.

Written order does not affect type checking, coordinate meaning, reference
evaluation, or legal implementation schedules. A serializer may choose any
stable presentation without making that presentation part of the language.

An expected result type **MUST NOT** introduce an axis or determine expression
inference. An optional binding annotation is checked only after the
right-hand-side type has been derived.

## 4. Modules and values

### 4.1 Program structure and name resolution

A program contains imports, axis declarations, global declarations, and
function declarations in that order. Module declarations are visible
throughout the module after the complete declaration set has been collected.

An identifier in an axis position resolves first to a static parameter of the
current function and then to a concrete module axis. Other identifiers resolve
lexically to the nearest permitted value or function binding. An unresolved
or multiply resolved name is a static error.

Names of the form `%N` are `perc-id` tokens reserved for TTL output. An
`at-id` is reserved for a compiler primitive or declared target function and
cannot conflict with an application identifier.

### 4.2 Imports and the target environment

An import makes a target vocabulary visible. `import ttl.lang;` selects the
portable vocabulary. A conforming environment **MUST** provide at least
`bool`, `i32`, `f16`, and `f32`, scalar arithmetic and comparisons, `exp`,
`log`, `log2`, `exp2`, `ceil`, `rsqrt`, explicit scalar conversions, and typed
selection.

The reference `ttl.lang` vocabulary also declares optional `f8e4m3fn` and
`f4e2m1` conversion functions for source compatibility with quantized models.
They are not part of the mandatory baseline. A module that references one is
valid only in an environment that implements the corresponding dtype and
target operation.

`+`, `minimum`, and `maximum` are reducers with declared identities. For every
reducer `r` admitted over an approximate element type, the environment used to
check or execute the module **MUST** define an order-invariant admissibility
predicate:

```text
admissible_r(M, observed)
```

`M` is the finite multiset of semantic input values and `observed` is the
runtime result. The predicate defines the permitted numerical approximation
without referring to an enumeration or grouping of `M`. It **MUST** specify
the treatment of every admitted special value, including NaNs, infinities,
signed zero, and overflow. Values outside its stated domain produce a target
numerical-domain error. For an exact element type, admissibility is equality
with the unique reduction in the target's declared scalar domain. Exact
overflow behavior is part of that domain and **MUST NOT** expose intermediate
grouping.

Distinct dtype spellings remain distinct; YTL performs no implicit
element-type conversion.

### 4.3 Global values

A `global` declaration introduces one immutable tensor value with
module-instance lifetime:

```ttl
axis layer, batch, head, channel, vocabulary, hidden, $past over batch;

global embedding: f16[vocabulary hidden];
global kv_cache: f16[layer batch $past head channel];
```

Its type **MUST** be complete and contain only concrete axes. A module instance
supplies exactly one well-typed value for each global before an export is
invoked. One invocation observes one immutable snapshot; the environment may
replace a global value between invocations.

YTL assigns no application role such as weight, parameter, constant, or cache
to a global. Elaboration preserves the declaration and direct references; it
does not add the global to an export parameter list.

### 4.4 Functions and static parameters

A function declaration binds its name at module scope. Its header binds static
parameters between `[` and `]` and runtime value parameters between `(` and
`)`. Its body is checked in that lexical environment and may refer directly to
module globals.

A static parameter is classified by its uses:

- use in an axis position makes it a generic axis;
- use in a static expression makes it a static scalar; and
- one parameter **MUST NOT** be used in both roles.

These are the only two roles. For example, `a` is a generic axis, while
`use_bias` is a static Boolean that controls elaboration:

```ttl
function identity[a](x: f32[a]) -> f32[a] := x;

function maybe_bias[use_bias](x: f32, bias: f32) -> f32 :=
    if use_bias then x + bias else x;
```

Static scalar parameters also supply integer extent expressions and structural
recursion counters. They disappear during elaboration just as generic axes do.

Passing a static parameter to another function imposes the corresponding
callee-parameter classification. All uses **MUST** imply one classification.
An unused static parameter is invalid.

Functions are first-order. An untyped helper parameter specializes to the
complete type of its actual argument.

A declaration without a body is permitted only for a `target` function.
Ordinary and exported functions **MUST** have bodies.

### 4.5 Function overloading

Several declarations may use the same function name when their signatures
differ. This permits operations with one conceptual role to retain one name
across element types, ranks, or axis signatures.

```ttl
function keep(x: f16) -> f16 := x;
function keep(x: f32) -> f32 := x;
```

`keep(x)` selects the declaration whose parameter type is the inferred type of
`x`; the expected result type cannot force either choice.

At a call, overload resolution considers static-argument arity and kind,
value-argument arity, and parameter compatibility after generic-axis
substitution and untyped-parameter specialization. Exactly one declaration
**MUST** match. Result types do not participate, and declaration order does not
break a tie. Zero or multiple matches are static errors.

### 4.6 Function markers and clauses

An `export` function is a module entry point. Its parameters and results
**MUST** have complete types containing concrete axes. It **MUST NOT** have
static parameters.

A `reducer` function is dyadic, rank-zero, and returns the same element type
as its operands. It declares exactly one scalar literal `identity`. The
function and identity **MUST** form a commutative monoid: combining values is
associative and commutative, and combining any value with the identity returns
that value. A reduction therefore denotes an invariant of its coordinate
domain, not a traversal or fold.

For exact element types these laws are exact. For approximate element types,
such as finite-precision floating point, they apply to the mathematical
operation; an observed result must satisfy the target environment's
order-invariant admissibility predicate. A reducer declaration asserts these
laws. An implementation may check the assertion but is not required to prove
it. Use of an approximate reducer for which the environment supplies no
admissibility predicate is a static error.

Subtraction, division, concatenation, left folds, right folds, and any other
order-dependent operation are not reducers. An ordered computation may be
provided by an explicitly ordered primitive, but **MUST NOT** be presented as
a reduction.

The syntax and role of `target` functions are defined with primitives in
Section 6.1.

### 4.7 Local values and binders

A value parameter is visible throughout its function body. A `let` binding
becomes visible only after its right-hand side and remains visible through the
rest of that `let`. A local binding **MUST NOT** shadow a global, parameter, or
earlier local binding.

The same `binder` production names a single result or destructures multiple
results:

```ttl
x = expression
(x, y) = expression
```

Every identifier in one binder **MUST** be distinct. If an annotation is
present, its result arity **MUST** equal the binder arity.

## 5. Core expression forms

This section defines the minimum expression system. Section 6 defines the
tensor primitives used by that system. Section 7 then defines convenient
surface notation that elaborates to these forms.

Every expression derives its result type from its syntax, operands, and
lexical environment. Unless stated otherwise, operands evaluate from left to
right and values are immutable.

### 5.1 Literals and references

An integer, floating-point, or Boolean literal is rank zero. A value reference
has the type of its binding. A function name is not a first-class value and
may occur only as a call or primitive clause reference.

### 5.2 Function calls

A call supplies one static argument for each static parameter and one value
argument for each value parameter. A generic-axis argument **MUST** resolve to
a concrete axis identity. A static scalar argument **MUST** evaluate during
elaboration.

A typed parameter matches an argument only when element type, rank, concrete
axis identities after substitution, and current static or dynamic domains
agree. Exactly one overload **MUST** match.

### 5.3 Sequential `let`

A sequential `let` evaluates one or more bindings in textual order and
returns its body:

```ttl
let
    y = f(x);
    z = g(y)
in
    z
```

Bindings are visible only to later bindings and the body. Their names do not
escape the expression.

### 5.4 Conditional expressions

For a runtime conditional, the condition has element type `bool`. Both
alternatives **MUST** have the same type. The condition may broadcast over
that type's coordinate domain but may not select a type, extent, or dynamic
descriptor. Reference evaluation eagerly computes both alternatives and
selects elementwise.

A conditional controlled entirely by static values is structural and follows
Section 9.2.

### 5.5 Primitive expressions

A primitive expression directly invokes a tensor primitive:

```text
@name[input-axis-patterns -> result-axis-pattern](arguments) clauses
```

Input axis patterns are checked against operand types at the call site. Result
types derive from operands, the axis specification, and static extents
resolved as defined in Section 3.4. The available primitives are defined in
Section 6.

### 5.6 Multiple results

Multiple results are permitted only in function result signatures, returned
expressions, and destructuring binders. They are not first-class tuple
tensors. Each component is typed independently, and a binder must have the
same arity as the value it destructures.

## 6. Tensor primitives

### 6.1 Target functions and primitive names

Every compiler-owned operation has an `@`-prefixed name. Tensor primitives
such as `@einsum` are defined directly by this manual. A target environment may
provide additional rank-zero scalar operations by declaring **target
functions**:

```ttl
target function @exp_f32(x: f32) -> f32;
target function @select_f32(c: bool, x: f32, y: f32) -> f32;
```

Syntactically, a target function has the `target` marker, an `@`-name, complete
rank-zero parameter and result types, and a terminating semicolon in place of
`= body`. It **MUST NOT** have static parameters or a body. Its implementation
is supplied by the imported target vocabulary. Application modules **MUST NOT**
redeclare a target function.

A target call has application syntax, for example `@exp_f32(x)`. It lifts the
declared scalar operation pointwise over the union of its operand axes. Shared
named axes align and absent axes broadcast. Its result dtype comes from the
target declaration, and its result axes are that union. A target declaration
is not a tensor primitive axis specification and cannot contain an axis
pattern. This direct application is retained in TTL; it is not merely surface
syntax.

### 6.2 `@einsum`

An einsum has one input axis pattern per operand and one result axis pattern:

```ttl
@einsum[row shared, shared column -> row column](a, b)
```

Each input axis pattern **MUST** name exactly the axes of its operand; position
does not establish alignment. Let the iteration axes be the set union of the
axes named by the input axis patterns. An axis occurring in an input and the
result is free. An axis occurring in an input but not the result is reduced.
Every result axis **MUST** occur in an input. Reordering axes within any axis
pattern does not change the expression.

The optional `map` clause names a scalar function receiving one element from
each operand in operand order. With one operand and no clause, mapping is
identity. With two or more operands and no clause, mapping is left-associated
`*`.

When reduced axes exist, `reduce` selects a reducer and defaults to `+`. A
reduce clause is forbidden when no axis is reduced. The reduced axes form a
set `K`. For each result coordinate `q`, mapping over every valid coordinate
assignment to `K` produces a finite multiset `M_q`. The result at `q` is the
unique commutative-monoid reduction of `M_q`; the empty multiset produces the
reducer identity. A multiset has no enumeration or parenthesization, so neither
is part of reduction semantics. An approximate observed result **MUST** satisfy
the admissibility predicate of Section 4.2.

### 6.3 `@gather`

```ttl
@gather[source -> inserted](data, indices)
```

The primitive replaces the named source axis of `data` with the named axes of
`indices`. Indices have element type `i32`. Shared index and data axes align
by identity. A negative or out-of-domain index is a dynamic error.

### 6.4 `@scatter`

Scatter is a functional, reducer-controlled indirect update:

```ttl
@scatter[token -> vocabulary](base, updates, indices) reduce=+
```

It has exactly three operands: `base`, `updates`, and `indices`. Its axis
specification has one contribution pattern before `->` and exactly one
destination axis after it. Every item is a bare axis. A `reduce` clause is
required and a `map` clause is forbidden. `base` and `updates` have one dtype,
`indices` has dtype `i32`, and the reducer combines two elements of the value
dtype into one element of that dtype.

Let `B`, `U`, and `J` be the axis sets of `base`, `updates`, and `indices`; let
`I` be the contribution-axis set named before `->`; let `s` be the destination
axis; and let `R = U minus I`. The following rules apply:

1. `s` occurs in `B` and occurs in neither `U` nor `J`.
2. `I` is a subset of both `U` and `J`.
3. `J` is a subset of `U`, and `J minus R = I`.
4. `B minus {s}` is a subset of `R`.
5. Aligned axis occurrences have compatible domains, and the inferred result
   type is well formed under Section 3.6.

The result has the value dtype and axis set `R union {s}`. The domain of `s`
comes from `base`; every other result domain comes from `updates`. `base`
broadcasts over axes in `R` that it omits, while `indices` broadcasts over
axes in `R` that it omits. Consequently, `base` supplies the complete static
or dynamic destination domain without an expected result type supplying
shape information.

For result coordinate `q`, let `C_q` be the multiset of update elements whose
coordinates agree with `q` on `R` and whose corresponding index equals
`q(s)`. The result element is the invariant reduction of the multiset

```text
{ base(project(q, B)) } multiset-union C_q
```

under the named reducer. Thus collisions have no writer order, and a
coordinate receiving no updates retains its `base` value. With the additive
reducer, a zero-valued base produces conventional scatter-add and supplies the
reverse-mode value operation corresponding to `@gather`; indices are ordinary
discrete inputs and are not differentiated by that correspondence. A negative
index or an index outside the destination domain at its retained coordinate is
a dynamic error.

Inside an axis scope, omitted prefix axes of `base` broadcast normally. For
example, an `f32[vocabulary]` base may be used with `f32[batch token]` updates
inside `let [batch]`; the result is `f32[batch vocabulary]`.

### 6.5 `@rearrange`

`@rearrange` performs a value-independent coordinate transformation. Every
valid axis specification **MUST** belong to exactly one of the two classes
below:
a single-source coordinate map or a multi-source concatenation. A
axis specification belonging to neither class is invalid. Its decomposition into
the class's constituent relations **MUST** be unique; an ambiguous
decomposition is invalid. Both classes must produce a contextually well-formed
result under Section 3.6.

#### 6.5.1 Single-source coordinate map

A single-source rearrangement has exactly one operand. Its axis specification
is a composition of disjoint local relations. Every input and result axis
occurrence, and every factor named by an axis-product annotation, **MUST**
belong to exactly one relation:

- A **preserved axis** has the same identity and domain on both sides.
- A **relabel relation** replaces one source axis with one distinct result
  axis having the same domain. It is written by the two unmatched bare axes,
  as in `@rearrange[token -> source](x)`. This is an explicit value
  transformation, not an alias declaration. Static relabels require equal
  extents. Dynamic relabels require equal support sets and transfer the source
  descriptor under the result identity. If unmatched axes admit more than one
  such pairing, the rearrangement is ambiguous and invalid.
- A **product relation** replaces one static parent axis `p` with an ordered
  sequence of distinct static factor axes `f1, ..., fn`, or performs the
  inverse replacement. It is written by annotating the parent pattern item, for
  example `p := [f1 * ... * fn]`. The factors occur as ordinary axes on the opposite
  side; factor names inside the annotation are relation metadata, not axes of
  the parent's side.
- A **unit removal** omits one input axis whose extent is one.
- A **unit insertion** introduces a fresh result axis item `a := 1`.
- A **replication** introduces a fresh result axis item `a := N`, where `N` resolves
  during elaboration to an integer greater than one.

The axes participating in different local relations **MUST** be disjoint.
Several independent relations may therefore be expressed in one
`@rearrange`. A source axis may be absent from the result only through a
relabel, product relation, or unit removal. A result axis may be absent from
the source only through a relabel, product relation, unit insertion, or
replication. A fresh
dynamic axis is forbidden because no source supplies its descriptor.

For a product relation with factor extents `n1 ... nn`, the checked extent
equation and coordinate map are:

```text
extent(p) = n_1 * ... * n_n
p = sum(i=1..n, f_i * product(j=i+1..n, n_j))
```

The last factor therefore varies fastest. Splitting computes `p` from the
result factor coordinates; merging uses the unique inverse Euclidean
decomposition of `p`. A product has at least two factors; a one-axis relation
would be forbidden aliasing rather than factorization.

Replication ignores the fresh coordinate when reading the source. For
example:

```ttl
@rearrange[item -> copy := 2 item](x)
```

denotes `result[copy, item] = x[item]`. Unit insertion and removal use the
unique zero coordinate of their axes. If all local relations are preserved
axes, the rearrangement is the identity; changing only their written order has
no effect, and elaboration **SHOULD** remove the expression.

#### 6.5.2 Multi-source concatenation

A multi-source concatenation has at least two operands and contains no product,
unit, or replication relation. There must be one common axis set `C`, one
distinct concatenated axis `s_i` for each operand `i`, and one fresh result
axis `r`, such that:

```text
axes(input_i) = C union {s_i}
axes(result)  = C union {r}
```

The `s_i` and `r` identities **MUST** be pairwise distinct. Every axis in `C`
has the same domain in every operand and the result. The `s_i` and `r` **MUST**
either all be static or all be dynamic. In the dynamic case they **MUST** have
the same support set. Their extents satisfy:

```text
extent(r, c) = sum(i, extent(s_i, c))
```

for each coordinate assignment `c` to the shared support set. At result
coordinate `r=k`, let `i` be the unique operand whose half-open prefix-sum
interval contains `k`; the result reads that operand at
`s_i = k - sum(j<i, extent(s_j, c))`. Operand order therefore orders
coordinates within the single result axis. It does not order independent
tensor axes.

For example:

```ttl
axis batch, channel,
     $past over batch, $step over batch, $source over batch;

@rearrange[
    batch $past channel,
    batch $step channel
    -> batch $source channel
](past, step)
```

Every relation established by `@rearrange` is local to that expression. After
the result is typed, no split, merge, unit, replication, or concatenation
provenance is retained. `@rearrange` **MUST NOT** gather, scatter, reduce,
apply a scalar operation, or otherwise discard coordinates.

## 7. Surface shorthand

The forms in this section add no primitive tensor operation. Elaboration
reduces them to the core expressions and primitives of Sections 5 and 6.

### 7.1 Unary and binary scalar operations

A scalar operation applied to tensors is lifted over the set union of their
axes. Shared axes align by identity. An operand missing an axis is
invariant along that axis and broadcasts without a physical singleton
dimension.

For scalar mapper `m`:

```text
axes(m(x0, ..., xn)) = union(axes(x0), ..., axes(xn))
```

Incompatible domains are a static error. There is no implicit element-type
conversion.

### 7.2 Map-reduce expressions

This form exists only in YTL 2.0:

```ttl
left reducer.mapper[reduction-axes] right
```

For example:

```ttl
A +.*[shared] B
```

The left name **MUST** resolve to a reducer and the right name to a dyadic
mapper. Let `I` be the set union of operand axes and `K` the set of distinct
bracketed axes. Every member of `K` **MUST** occur in `I`. The result axes are
the set difference `I minus K`.

An operand omitting a reduction axis broadcasts along it before mapping.
Bracket order has no semantic effect: `r.m[a b]` and `r.m[b a]` are the same
operation. The expression elaborates to one `@einsum` and does not materialize
the mapped tensor first.

### 7.3 Selectors

A selector indexes or slices a static axis by identity, regardless of where
that axis is written in the operand type:

```ttl
axis batch := 2, token := 8, hidden := 4;

function inspect(x: f32[batch token hidden]) :=
    let
        one = x[token=3];       // f32[batch hidden]
        four = x[token=2:6]     // f32[batch token := 4 hidden]
    in
        (one, four);
```

For `x[a=i]`, `a` **MUST** occur exactly once in the inferred type of `x` and
`i` **MUST** be a static integer in its domain. The result type is the type of
`x` with axis identity `a` removed; every other named axis and its domain are
unchanged.

For `x[a=begin:end]`, `a` has the same requirement and both bounds **MUST** be
static. The result retains identity `a` with current extent `end - begin` and
is reindexed from zero: result coordinate `a=j` contains the source value at
`a=begin+j`. The source offset is part of the selector operation, not the
result type. Thus selection neither infers by position nor invents a new axis
identity. Multiple selections apply these rules to distinct named axes.
Bounds **MUST** be statically valid. Selecting a dynamic axis is invalid in YTL
2.0.

Consequently, `x[token=2:6]` and `x[token=4:8]` both have the axis entry
`token := 4`. They align by their new coordinates `0 .. 3`; their different
source offsets have already been consumed by the two selector operations.

## 8. Scoped computation

Scoped computation is the second YTL 2.0 addition. It factors a common named
coordinate environment, conventionally called a prefix, out of a group of
ordinary tensor expressions. This section adds no new tensor value or
primitive; Section 9.4 defines its flattening.

### 8.1 Scope syntax and prefix

```ttl
axis batch, head, token, feature;

function add_bias(
    x: f32[batch head token feature],
    bias: f32[head feature]
) -> f32[batch head token feature] :=
let [batch head token]
    y = x + bias
in
    y;
```

The `let [batch head token]` header establishes one coordinate for each named
axis while its body is evaluated. These named coordinates form this scope's
local prefix. A prefix is a finite mapping from axis identities to coordinates,
not a coordinate tuple. In a nested scope, the effective prefix is the
disjoint union of the enclosing and local prefixes. Reordering a scope header
does not change the scope.

Header axes **MUST** be distinct and **MUST NOT** repeat an enclosing prefix
axis. The extent of an axis written without an extent constraint **MUST** be
resolvable during elaboration under Section 3.4. A fresh static axis must carry
a positive extent constraint, as in `copy := 2`. A scope cannot introduce a fresh
dynamic axis. The support set of every dynamic header axis
**MUST** be a subset of the effective prefix.

Bindings follow the sequential rules of Section 5.3. A scoped `let` may have
no bindings because its body alone may require lifting.

### 8.2 Fibre selection

The code in Section 8.1 is the YTL source. The following table is its semantic
derivation, not additional source syntax. At the prefix coordinate where
`batch=b`, `head=h`, and `token=t`:

| Reference | Coordinates fixed by the scope | Type visible inside the scope |
|---|---|---|
| `x` | `batch=b, head=h, token=t` | `f32[feature]` |
| `bias` | `head=h` | `f32[feature]` |

`bias` therefore varies with `head` but is invariant with respect to `batch`
and `token`. The local addition combines two `f32[feature]` fibres. No
`fibre(...)` function or runtime selector is added to YTL.

Let `P` be the effective prefix and let an outer tensor have axes `A`. At one
coordinate of `P`, a reference fixes the coordinates in `P intersect A` and
exposes the remaining axes `A minus P`.

For example, under prefix `[batch head token]`:

| Outer axes | Fibre axes | Prefix axes along which the value is invariant |
|---|---|---|
| `[feature]` | `[feature]` | `batch head token` |
| `[batch feature]` | `[feature]` | `head token` |
| `[head feature]` | `[feature]` | `batch token` |
| `[batch token feature]` | `[feature]` | `head` |
| `[batch head token feature]` | `[feature]` | none |

Selection denotes a view of the value at the current prefix coordinate. It
does not require copying or allocation.

### 8.3 Result lifting

If the body has fibre type `T[F]`, the scoped expression has type
`T[P union F]` relative to its enclosing scope:

```text
Lift_P(T[F]) = T[P union F]
```

Nested lifting composes by disjoint set union. `F` **MUST NOT** contain an axis
already in `P`. The complete prefix remains in the result even when the body
is invariant along part of it. For a dynamic fibre whose descriptor was
partially applied by `P`, lifting reassembles those fibre descriptors over the
corresponding coordinates of `P`.

### 8.4 Calls within a scope

Tensor arguments are viewed as their current fibres before function matching.
The returned fibre is lifted by the caller's scope. The callee does not
receive the caller's prefix and cannot observe neighbouring prefix
coordinates.

More precisely, for effective prefix `P`, an argument whose outer axes are
`A` participates in overload resolution with visible axes `A minus P`. A
typed parameter **MUST** have exactly those visible axes after generic-axis
substitution. A candidate parameter that names an axis in `P` is not viable,
even if the underlying flat argument carries that axis. Result types are
checked by the same rule before the caller restores `P`.

Consequently, an operation that consumes a prefix axis must occur in an
ancestor scope where that axis is still present in the fibre type.

### 8.5 Scope restrictions and local lifetime

Within prefix `P`:

- no expression may reduce an axis in `P`;
- no local result may introduce another occurrence of an axis in `P`;
- no call may observe neighbouring coordinates along `P`; and
- cross-prefix communication requires an explicit operation outside the
  scope.

Lifting defines a value, not required storage. A fibre-local intermediate may
remain a scalar, register fragment, or other local representation when the
observable result is unchanged.

## 9. Finite elaboration

Elaboration of every accepted YTL program **MUST** terminate and produce a
finite TTL module before tensor execution begins. Runtime tensor values and
dynamic-axis extents **MUST NOT** determine which functions, bindings, or
primitive nodes are emitted. The resulting graph may be large, but it is
finite and completely explicit.

This is a structural finiteness guarantee, not a numerical determinism
guarantee. It does not prescribe tensor traversal, reduction grouping, or one
bit pattern for approximate floating-point results.

### 9.1 Static parameters

A static parameter used in a static expression denotes a non-negative `i32`
unless its uses require `bool`. Static arguments and extent expressions are
evaluated before tensor execution. They are not runtime tensors.

### 9.2 Structural conditionals and recursion

A static conditional elaborates only its selected branch. Recursion is
permitted only through structurally decreasing static specializations. Every
recursive edge **MUST** preserve or decrease corresponding static arguments,
and at least one argument **MUST** decrease by a positive integer. Tensor
values cannot control structural recursion. Because each function body has
finitely many calls, these requirements make every reachable specialization
graph finite.

### 9.3 Specialization and shorthand expansion

For every reachable call, elaboration binds static arguments, resolves one
overload, verifies axis substitution and extent constraints, and specializes
untyped parameters. Ordinary functions and scalar helpers are inlined.

Unary and binary tensor operations become direct pointwise target
applications. Map-reduce expressions become `@einsum`. Selectors become explicit coordinate
projections with resolved result extents. No shorthand remains in TTL.

### 9.4 Scope flattening

At effective prefix `P`, a local binding with fibre axes `F` receives flat
axes `P union F`. References to outer tensors explicitly select coordinates
`P intersect axes(tensor)`. Missing prefix axes become explicit broadcasts.

The body is lifted over all of `P`. A missing static prefix axis whose extent
has been resolved is introduced by `@rearrange`. A dynamic prefix axis must be
preserved through an operand carrying its descriptor. No author scope remains
in TTL.

### 9.5 Required output

For every export, elaboration **MUST**, in dependency order:

1. resolve imports and module declarations;
2. classify and bind static parameters;
3. verify axis identities and their static or dynamic domains;
4. evaluate structural conditionals and unfold structural recursion;
5. resolve overloads and inline reachable functions;
6. expand surface shorthand;
7. flatten scoped computation;
8. resolve primitive coordinate relations;
9. make maps, reducers, broadcasts, and scalar conversions explicit;
10. split compound scalar expressions into target-atomic tensor bindings; and
11. emit resolved concrete axis declarations, including the dynamic-axis
    dependency partial order, globals, and finite primitive-only export graphs.

No helper, overload set, static parameter, generic axis, recursion, ordinary
call, shorthand expression, scope, or untyped binding remains in TTL.
Elaboration **MUST** preserve the reference result and exported signatures.

Two elaborated TTL modules are structurally equivalent when they have the same
concrete axis identities, dynamic-axis dependency partial order, domains,
globals, export signatures, and isomorphic primitive dependency graphs, modulo
generated binding names and the textual enumeration order of unordered axis
sets. Operand order, multiple-result order, concatenation operand order, and
axis-product factor order remain part of that graph.

## 10. Reference evaluation

### 10.1 Module environment and evaluation

Evaluation begins with one well-typed value for every global. Each export
invocation observes one immutable snapshot. Function arguments and expression
operands evaluate from left to right. Values are immutable. Tensor coordinates
may be visited in any order because coordinate traversal is not observable.

A reduction denotes the finite-multiset operation of Section 6.2. For exact
element types it has one mathematical value. For approximate element types, a
conforming observed result **MUST** satisfy the environment's admissibility
predicate. A reference implementation may choose a fixed traversal for
repeatability, but that choice is not YTL semantics and a program **MUST NOT**
depend on it. Unless a target's admissibility predicate itself requires one,
YTL does not require bitwise-identical approximate results across conforming
implementations.

A scoped expression is defined by its flattened result. An interpreter may
evaluate fibres directly when the observable result is identical.

### 10.2 Required diagnostics

A conforming implementation **MUST** reject:

- lexical or grammatical invalidity;
- duplicate, unresolved, or illegally shadowed bindings;
- an undeclared concrete axis or inconsistently classified static parameter;
- an unresolved or ambiguous call;
- an axis mapping or scope prefix that fails contextual well-formedness;
- incompatible types, axis identities, or static or dynamic domains;
- use of an approximate reducer without an admissibility predicate;
- an unresolved static scope or rearrange-introduced axis;
- reduction across a scope prefix;
- invalid structural recursion;
- primitive arity, pattern, mapper, reducer, result, or rearrange-classification
  errors; and
- invalid selector constraints.

Out-of-domain gather or scatter indices and scalar numerical-domain failures
are dynamic errors. Silent positional axis matching, semantic renaming,
truncation, padding, wrapping, or clamping is forbidden unless a target
operation explicitly defines it.

## 11. Conformance and compatibility

### 11.1 YTL conformance

A YTL 2.0 implementation may provide parsing, static checking, elaboration,
reference evaluation, or any stated combination. It **MUST** identify the
parts it implements. Each implemented part **MUST** obey this manual.

### 11.2 YTL 1.0 profile

The [YTL 1.0 compatibility profile](ytl-lrm-1.0.md) uses the common grammar
and all common semantics in this manual. It excludes only `scope-expr` and
`map-reduce-op`, as shown in Section 2.3. Every YTL 1.0 program is therefore a
YTL 2.0 program with the same static meaning, result, and structurally
equivalent TTL output.

### 11.3 TTL relationship

The [TTL 1.0 specification](ttl-lrm-1.0.md) is permanently a strict syntactic subset of
YTL. It contains resolved concrete axis declarations, including the
dynamic-axis dependency partial order, complete globals and export signatures,
`%N` bindings, compiler primitives, and fixed scalar target references. Runtime
values carry dynamic descriptors.

Every TTL grammar nonterminal occurs in the common YTL grammar. TTL obtains
each construct only by restricting its corresponding right-hand side.
Consequently, a grammar simplification applicable to a shared construct
belongs in the common YTL grammar as well as in TTL.

TTL contains no imports, helpers, overloads, static parameters, generic
axes, structural conditionals, recursion, ordinary calls, shorthand,
author scopes, or untyped bindings. YTL 1.0 and YTL 2.0 elaborate to the same
TTL 1.0 language.

## Appendix A. Complete examples

### A.1 Scoped softmax

```ttl
axis batch, token, class;

function softmax[a](x: f32[a]) -> f32[a] :=
    let
        m = @einsum[a ->](x) reduce=maximum;
        e = exp(x - m);
        z = @einsum[a ->](e) reduce=+
    in
        e / z;

export function normalize(
    logits: f32[batch token class]
) -> f32[batch token class] :=
    let [batch token] in
        softmax[class](logits);
```

Inside the scope, `logits` has fibre `f32[class]`. Scope exit restores the
prefix `[batch token]`.

### A.2 Scoped attention

```ttl
axis batch, head, query, key, channel, value;

function softmax[a](x: f32[a]) -> f32[a] :=
    let
        m = @einsum[a ->](x) reduce=maximum;
        e = exp(x - m);
        z = @einsum[a ->](e) reduce=+
    in
        e / z;

export function attention(
    q: f32[batch head query channel],
    k: f32[batch head key channel],
    v: f32[batch head key value]
) -> f32[batch head query value] :=
    let [batch head query]
        score = q +.*[channel] k;
        weight = softmax[key](score)
    in
        weight +.*[key] v;
```

Inside the scope, `q`, `k`, and `v` have fibre axes `[channel]`,
`[key channel]`, and `[key value]`. The two map-reduce expressions consume
`channel` and `key`; lifting restores `[batch head query]`.

### A.3 Whole-array duplication

```ttl
axis item, copy := 2, doubled;

export function duplicate(
    x: f32[item]
) -> f32[doubled := 2 * item] :=
    let
        tiled = let [copy] in x
    in
        @rearrange[copy item -> doubled := [copy * item]](tiled);
```

The scope lifts `x` to `[copy item]`. The rearrange merges that product into
`doubled`, producing two consecutive complete copies.

## Appendix B. Related work and naming

This section is non-normative. It records intellectual influences and naming
rationale; it does not add syntax or semantics.

### B.1 Yang, gauge symmetry, and the language names

**YTL** expands to **Yang Tensor Language** and is named in honour of the
physicist Chen-Ning (Zhenning) Yang. In their groundbreaking 1954 paper,
[“Conservation of Isotopic Spin and Isotopic Gauge Invariance”](https://doi.org/10.1103/PhysRev.96.191),
Yang and Robert Mills formulated local isotopic gauge invariance and the
associated nonlinear field equations. The work became the foundation of
Yang-Mills gauge theory.

The name honours Yang's work on Gauge Symmetry and suits a language concerned
with exposing mathematical structure and invariant coordinate meaning. It
does not imply that YTL implements gauge theory or derives technical semantics
from it.

**TTL** expands to **Tiny Tensor Language**. It is a small, primitive-only
subset of YTL that retains the complete tensor semantics of the language. TTL
is the elaboration target of YTL and is intended as the input to interpreters
and lowering compilers.

### B.2 APL and NESL

Kenneth E. Iverson's [*A Programming Language*](https://www.jsoftware.com/papers/APL1.htm)
and the APL tradition are long-term influences on YTL's array-first design
sensibility. APL demonstrated the value of treating whole arrays as the normal
objects of computation and of building substantial algorithms from a compact,
compositional operator vocabulary. YTL's APL-style infix map-reduce notation
reflects that influence. YTL does not adopt APL's broad symbolic vocabulary or
implicit rank reasoning; it keeps a small operator surface and makes semantic
axis names explicit so that expressions remain locally explainable.

[NESL](https://www.cs.cmu.edu/~scandal/nesl.html) is a second long-term
influence, specifically through its treatment of irregular arrays. NESL's
implementation flattened nested sequences into value vectors accompanied by
[segment descriptors](https://www.cs.cmu.edu/~scandal/papers/nesl-ppopp93.html)
that record the length of each logical subsequence. This model motivates YTL's
decision to retain dynamic axes: a dynamic-axis descriptor records one runtime
extent for every coordinate assignment over its support set while the values
remain part of a tensor.

The two languages obtain nested parallel structure differently. NESL permits
nested sequence operations and recursive parallel calls, as used in
divide-and-conquer algorithms. YTL represents nesting through named axes and
axis scopes. Its recursion is structural, terminates during elaboration, and
does not provide runtime nested parallelism. YTL therefore adopts the
segmented-array insight without adopting NESL's recursive nested-sequence
execution model.

### B.3 Einstein notation, `einsum`, and Einops

YTL's tensor-expression design follows the tradition of the Einstein summation
convention. In library practice, [NumPy's `einsum`](https://numpy.org/doc/stable/reference/generated/numpy.einsum.html)
established a compact interface in which labelled input patterns and an
explicit `->` result pattern express contraction, reduction, diagonal
selection, broadcasting, and permutation.

[Einops](https://einops.rocks/) subsequently improved this notation with
readable, full-word axis names and applied related input-to-output patterns to
rearrangement, reshaping, repetition, and reduction. YTL's `@einsum` syntax is
directly inspired by the Einops improvement of `einsum`, and its `@rearrange`
syntax follows the same pattern-oriented approach. YTL promotes those library
patterns into language syntax, resolves their names as declared or generic
semantic axis identities, permits independently selected scalar mappers and
reducers, and elaborates the expressions to a target-independent primitive
core. The design lineage and motivation are described by the
[Einops paper](https://openreview.net/forum?id=oapKSVM2bcj).

### B.4 Einlang

[Einlang](https://github.com/einlang/einlang) and its book
[The Name in the Bracket](https://einlang.github.io/einlang/book/) make the
case that coordinate names must be recorded and checked rather than left as
positional knowledge or comments. Einlang's emphasis on semantic coordinates
strongly influenced YTL's decision to make axis identity part of tensor types.
Both languages aim to reject semantically swapped axes even when their extents
agree.

The designs nevertheless make different choices. YTL declares concrete axis
identities at module scope and uses static parameters as generic axis
placeholders. It does not use output-index bindings to supply expression
types; the right-hand side and lexical scope determine the result. Its minimum
tensor algebra remains explicit `@einsum`, `@gather`, `@scatter`, and
`@rearrange`, while scoped fibres and infix map-reduce are optional authoring
forms. YTL scopes are flattened, ordinary functions are inlined, and the
result is the separate primitive-only TTL language.

YTL deliberately discards several mechanisms that would weaken this model:

- positional axis addressing or matching;
- expression-local labels as the source of concrete axis identity;
- output-index bindings as a source of expression types;
- implicit axis aliasing or semantic renaming;
- persistent source-axis lineage after an explicit `@rearrange` operation;
- runtime recursion or recurrence.

YTL's decisive design boundary is finite elaboration.
Structural recursion must terminate during elaboration; it never appears in
TTL or at runtime. Runtime values may supply tensor elements and dynamic-axis
extents, but they cannot decide the structure of the emitted computation.
Every accepted source program therefore closes into a finite, explicit
primitive graph before tensor execution begins.

This is a finiteness contract, not an evaluation-determinism contract. YTL does
not use elaboration to prescribe coordinate traversal, reduction grouping, or
a unique floating-point bit pattern. These choices retain Einlang's central
insight about semantic axes while deliberately placing YTL on the finite side
of the boundary between compile-time composition and runtime computation.

### B.5 Personal note from the author

Computer science has left us an extraordinary treasury of intellectual gems;
few, if any, of the ideas in YTL are wholly new. The choice and composition of
those gems are subjective, but the contribution sought here is mostly
subtractive: to reduce proven ideas until they form one canonical, coherent,
and minimal language. The practical hope is that engineers working at the
frontier of GPUs, FPGAs, and future accelerators can spend less time
pre-committing to implementation decisions, more time expressing the
computation they actually intend, and find greater satisfaction in the
craft of expression and the pursuit of invariants.
