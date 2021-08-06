# Autograd Mechanics

## 1. How Autograd encodes the History
```Autograd``` is a reverse differentiation system. Conceptually, autograd records a graph recording all of the operations that created the data as we execute operations, giving an ```acyclic direct graph``` whose leaves are the input tensors and roots are the output tensors. By `tracing` this graph from roots to leaves, we can automatically compute the gradients using `chain rule`.

Internally, ```autograd``` represents this graph as graph of ```Function``` objects, which can be ```apply()``` to compute the result of evaluating the graph. When computing the ```forward()```, autograd simontaneously performs the requested computations and builds up a graph representing the function that computes the `gradient` (the ```.grad_fn``` attribute of each ```torch.Tensor``` is an entry point into this graph). When the forwards pass is completed, we evaluate this graph in the backwards pass to compute the gradients.

## 2. Locally Disabling gradient Computation
There are several mechanisms available from Python to locally disable gradient computation:

To disable gradients across entire blocks of code, there are context managers like no-grad mode and inference mode. For more fine-grained exclusion of subgraphs from gradient computation, there is setting the ```requires_grad``` field of a tensor.

Below, in addition to discussing the mechanisms above, we also describe evaluation mode ```(nn.Module.eval())```, a method that is not actually used to disable gradient computation but, because of its name, is often mixed up with the three.

## 3. Setting ```requires_grad```
```requires_grad``` is a flag that allows for fine-grained exclusion of subgraphs from gradient computation. It takes effect in both the forward and backward pass.

During forward pass, an operation is only recorded in the backward graph if at least one of its input tensors require grad. During the backward pass (```.backward()```), only leaf tensors with ```requires_grad=True``` will have gradients accumulated into their ```.grad``` fields.

It is important to note that even though every tensor has this flag, setting it only makes sense for leaf tensors (tensors that do not have a `grad_fn`, e.g., a `nn.Module`’s parameters). Non-leaf tensors (tensors that do have `grad_fn`) are tensors that have a backward graph associated with them. Thus their gradients will be needed as an intermediary result to compute the gradient for a leaf tensor that requires grad. From this definition, it is clear that all non-leaf tensors will automatically have `require_grad=True`.

Setting `requires_grad` should be the main way you control which parts of the model are part of the gradient computation, for example, if you need to freeze parts of your pretrained model during model fine-tuning.

To freeze parts of your model, simply apply `.requires_grad_(False)` to the parameters that you don’t want updated. And as described above, since computations that use these parameters as inputs would not be recorded in the forward pass, they won’t have their `.grad` fields updated in the backward pass because they won’t be part of the backward graph in the first place, as desired.

Because this is such a common pattern, requires_grad can also be set at the module level with `nn.Module.requires_grad_()`. When applied to a module, `.requires_grad_()` takes effect on all of the module’s parameters (which have `requires_grad=True` by default).

## 4. Grad Modes

Apart from setting `requires_grad` there are also three possible modes enableable from Python that can affect how computations in PyTorch are processed by autograd internally: default mode (grad mode), no-grad mode, and inference mode, all of which can be togglable via context managers and decorators.

## 4(a). Default Mode (Grad Mode)
The “default mode” is actually the mode we are implicitly in when no other modes like no-grad and inference mode are enabled. To be contrasted with “no-grad mode” the default mode is also sometimes called “grad mode”.

The most important thing to know about the default mode is that it is the only mode in which `requires_grad` takes effect. `requires_grad` is always overridden to be False in both the two other modes.

## 4(b). No-grad Mode (Grad Mode)

Computations in no-grad mode behave as if none of the inputs require grad. In other words, computations in no-grad mode are never recorded in the backward graph even if there are inputs that have ```require_grad=True```.

Enable no-grad mode when you need to perform operations that should not be recorded by autograd, but you’d still like to use the outputs of these computations in grad mode later. This context manager makes it convenient to disable gradients for a block of code or function without having to temporarily set tensors to have ```requires_grad=False```, and then back to `True`.

For example, no-grad mode might be useful when writing an ```optimizer```: when performing the training update you’d like to update parameters in-place without the update being recorded by autograd. You also intend to use the updated parameters for computations in grad mode in the next forward pass.

## 4 (c). Inference Mode
Inference mode is the extreme version of no-grad mode. Just like in no-grad mode, computations in inference mode are not recorded in the backward graph, but enabling inference mode will allow PyTorch to speed up your model even more. This better runtime comes with a drawback: tensors created in inference mode will not be able to be used in computations to be recorded by autograd after exiting inference mode.

Enable inference mode when you are performing computations that don’t need to be recorded in the backward graph, AND you don’t plan on using the tensors created in inference mode in any computation that is to be recorded by autograd later.

It is recommended that you try out inference mode in the parts of your code that do not require autograd tracking (e.g., data processing and model evaluation). If it works out of the box for your use case it’s a free performance win. If you run into errors after enabling inference mode, check that you are not using tensors created in inference mode in computations that are recorded by autograd after exiting inference mode. If you cannot avoid such use in your case, you can always switch back to no-grad mode.

## 4(d). Evaluation Mode (```nn.Module.eval()```)
Evaluation mode is not actually a mechanism to locally disable gradient computation. It is included here anyway because it is sometimes confused to be such a mechanism.

Functionally, ```module.eval()``` (or equivalently `module.train()`) are completely orthogonal to `no-grad mode and inference mode`. How `model.eval()` affects your model depends entirely on the specific modules used in your model and whether they define any training-mode specific behavior.

You are responsible for calling model.eval() and model.train() if your model relies on modules such as torch.nn.Dropout and torch.nn.BatchNorm2d that may behave differently depending on training mode, for example, to avoid updating your BatchNorm running statistics on validation data.

It is recommended that you always use model.train() when training and model.eval() when evaluating your model (validation/testing) even if you aren’t sure your model has training-mode specific behavior, because a module you are using might be updated to behave differently in training and eval modes.

## 5.In-place operations with Autograd
Supporting in-place operations in autograd is a hard matter, and we discourage their use in most cases. Autograd’s aggressive buffer freeing and reuse makes it very efficient and there are very few occasions when in-place operations actually lower memory usage by any significant amount. Unless you’re operating under heavy memory pressure, you might never need to use them.

There are two main reasons that limit the applicability of in-place operations:
1. In-place operations can potentially overwrite values required to compute gradients.
2. Every in-place operation actually requires the implementation to rewrite the computational graph. Out-of-place versions simply allocate new objects and keep references to the old graph, while in-place operations, require changing the creator of all inputs to the Function representing this operation. This can be tricky, especially if there are many Tensors that reference the same storage (e.g. created by indexing or transposing), and in-place functions will actually raise an error if the storage of modified inputs is referenced by any other ```Tensor```.

### 6. In-place correctness checks
Every tensor keeps a version counter, that is incremented every time it is marked dirty in any operation. When a Function saves any tensors for backward, a version counter of their containing Tensor is saved as well. Once you access ```self.saved_tensors``` it is checked, and if it is greater than the saved value an error is raised. This ensures that if you’re using in-place functions and not seeing any errors, you can be sure that the computed gradients are correct.

## 7. Mutlithread Autograd
The autograd engine is responsible for running all the backward operations necessary to compute the backward pass. This section will describe all the details that can help you make the best use of it in a multithreaded environment.(this is relevant only for PyTorch 1.6+ as the behavior in previous version was different).

## 8. Concurrency on CPU
When you run `backward()` or `grad()` via python or C++ API in multiple threads on CPU, you are expecting to see extra concurrency instead of serializing all the backward calls in a specific order during execution. 

## 9. Non-Determinism
If you are calling `backward()` on multiple thread concurrently but with shared inputs (i.e. Hogwild CPU training). Since parameters are automatically shared across threads, gradient accumulation might become non-deterministic on backward calls across threads, because two backward calls might access and try to accumulate the same .grad attribute. This is technically not safe, and it might result in racing condition and the result might be invalid to use.

But this is expected pattern if you are using the multithreading approach to drive the whole training process but using shared parameters, user who use multithreading should have the threading model in mind and should expect this to happen. User could use the functional API `torch.autograd.grad()` to calculate the gradients instead of `backward()` to avoid non-determinism.

## 10. Graph Retaining 
If part of the autograd graph is shared between threads, i.e. run first part of forward single thread, then run second part in multiple threads, then the first part of graph is shared. In this case different threads execute `grad()` or `backward()` on the same graph might have issue of destroying the graph on the fly of one thread, and the other thread will crash in this case. Autograd will error out to the user similar to what call `backward()` twice with out `retain_graph=True`, and let the user know they should use `retain_graph=True`.

## 11. Thread Safety on Autograd Node
Since Autograd allows the caller thread to drive its backward execution for potential parallelism, it’s important that we ensure thread safety on CPU with parallel backwards that share part/whole of the GraphTask.

Custom Python `autograd.function` is automatically thread safe because of GIL. for built-in C++ Autograd Nodes(e.g. AccumulateGrad, CopySlices) and custom ```autograd::Function```, the Autograd Engine uses thread mutex locking to protect thread safety on autograd Nodes that might have state write/read.

## 12. No Thread Safety on C++ Hooks
Autograd relies on the user to write thread safe C++ hooks. If you want the hook to be correctly applied in multithreading environment, you will need to write proper thread locking code to ensure the hooks are thread safe

## 13. Autograd for Complex Numbers

The short version:
- When we  PyTorch to differentiate any function `F(z)` with complex domain and/or codomain, the gradients are computed under the  assumption that the function is a part of a larger real-valued loss function g(input)=Lg(input)=Lg(input)=L. The gradient computed is ```∂L/∂z∗```​ (note the conjugation of z), the negative of which is precisely the direction of steepest descent used in `Gradient Descent algorithm`. Thus, all the existing optimizers work out of the box with complex parameters. 
- This convention matches TensorFlow’s convention for complex differentiation, but is different from JAX (which computes `∂L/∂z`).
- If you have a real-to-real function which internally uses complex operations, the convention here doesn’t matter: you will always get the same result that you would have gotten if it had been implemented with only real operations.

## 14. What are Complex Derivatives ?
The mathematical definition of complex-differentiability takes the limit definition of a derivative and generalizes it to operate on complex numbers. Consider a function `f:C→Cf: ℂ → ℂf:C→C`

                        ‘f(z = x + yj) = u(x, y)+ v(x, y)j‘
where `u` and `v` are two variable real valued functions.

Using the derivative definition, we can write:

![Equation](Image.png, "Last Equation")