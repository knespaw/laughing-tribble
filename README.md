## Building Project

```bash
cd build
cmake ..
cmake --build .
```

## Dependencies Management

### 1. Installing `llama.cpp`

#### Cloning the repository

```bash
cd dependencies
git submodule add git@github.com:ggml-org/llama.cpp.git
```

#### Applying fix

Inside `dependencies/llama.cpp/src/models/t5.cpp`, at the top:

```c++
template <> llama_model_t5::graph<false>::graph(const llama_model & model, const llm_graph_params & params);
template <> llama_model_t5::graph<true>::graph(const llama_model & model, const llm_graph_params & params);
```

## Formatting

```bash
find . -name "*.cpp" -o -name "*.h" | grep -v "dependencies" | xargs clang-format -i -style=file
```
