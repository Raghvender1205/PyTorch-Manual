# Automatic Mixed Precision
```Automatic Mixed Precision Training``` is training with ```torch.cuda.amp.autocast``` and ```torch.cuda.amp.GradScaler``` together.

Instances of ```torch.cuda.amp.autocast``` enable autocasting for chosen regions. Autocasting automatically chooses the precision for GPU operations to improve performance while maintaining accuracy.

Instances of ```torch.cuda.amp.GradScaler``` help perform the steps of gradient scaling conviniently. Gradient Scaling improves convergence for networks with ```float16``` gradients by minimizing ```gradient underflow```.

## 1. Typical Mixed Precision Training

```python
# Creates model and optimizer in default precision
model = Net().cuda()
optimizer = optim.SGD(model.parameters(), ...)

# Creates a GradScaler once at the beginning of training.
scaler = GradScaler()

for epoch in epochs:
    for input, target in data:
        optimizer.zero_grad()

        # Runs the forward pass with autocasting.
        with autocast():
            output = model(input)
            loss = loss_fn(output, target)

        # Scales loss.  Calls backward() on scaled loss to create scaled gradients.
        # Backward passes under autocast are not recommended.
        # Backward ops run in the same dtype autocast chose for corresponding forward ops.
        scaler.scale(loss).backward()

        # scaler.step() first unscales the gradients of the optimizer's assigned params.
        # If these gradients do not contain infs or NaNs, optimizer.step() is then called,
        # otherwise, optimizer.step() is skipped.
        scaler.step(optimizer)

        # Updates the scale for next iteration.
        scaler.update()
```

## 2. Working with Unscaled Gradients
All gradients produced by ```scaler.scale(loss).backward()``` are scaled. If you wish to modify or inspect the parameters’ ```.grad``` attributes between ```backward()``` and ```scaler.step(optimizer)```, you should unscale them first. 

 For example, gradient clipping manipulates a set of gradients such that their global norm (see ```torch.nn.utils.clip_grad_norm_()```) or maximum magnitude (see ```torch.nn.utils.clip_grad_value_()```) is <=<= some user-imposed threshold. If you attempted to clip without `unscaling`, the gradients’ norm/maximum magnitude would also be scaled, so your requested threshold (which was meant to be the threshold for unscaled gradients) would be invalid.
 
```scaler.unscale_(optimizer)``` unscales gradients held by ```optimizer’s``` assigned parameters. If your model or models contain other parameters that were assigned to another optimizer (say ```optimizer2```), you may call ```scaler.unscale_(optimizer2)``` separately to unscale those parameters’ gradients as well.

## Gradient Clipping 

Calling ```scaler.unscale_(optimizer)``` before clipping enables you to clip unscaled gradients as usual.

```python

scaler = GradScaler()

for epoch in epochs:
    for input, target in data:
        optimizer.zero_grad()
        with autocast():
            output = model(input)
            loss = loss_fn(output, target)
        scaler.scale(loss).backward()

        # Unscales the gradients of optimizer's assigned params in-place
        scaler.unscale_(optimizer)

        # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        # optimizer's gradients are already unscaled, so scaler.step does not unscale them,
        # although it still skips optimizer.step() if the gradients contain infs or NaNs.
        scaler.step(optimizer)

        # Updates the scale for next iteration.
        scaler.update()
```
scaler records that ```scaler.unscale_(optimizer)``` was already called for this optimizer this iteration, so ```scaler.step(optimizer)``` knows not to redundantly unscale gradients before (internally) calling ```optimizer.step().```

## 3. Working with Scaled Gradients

### Gradient Accumulation
Gradient accumulation adds gradients over an effective batch of size ```batch_per_iter * iters_to_accumulate``` (* num_procs if distributed). The scale should be calibrated for the effective batch, which means ```inf/NaN``` checking, step skipping if ```inf/NaN``` grads are found, and scale updates should occur at effective-batch granularity. Also, grads should remain scaled, and the scale factor should remain constant, while grads for a given effective batch are accumulated. If grads are unscaled (or the scale factor changes) before accumulation is complete, the next backward pass will add scaled grads to unscaled grads (or grads scaled by a different factor) after which it’s impossible to recover the accumulated unscaled grads step must apply.

Therefore, if you want to ```unscale_``` grads (e.g., to allow clipping unscaled grads), call ```unscale_``` just before step, after all (scaled) grads for the upcoming step have been accumulated. Also, only call update at the end of iterations where you called step for a full effective batch:
```python
scaler = GradScaler()

for epoch in epochs:
    for i, (input, target) in enumerate(data):
        with autocast():
            output = model(input)
            loss = loss_fn(output, target)
            loss = loss / iters_to_accumulate

        # Accumulates scaled gradients.
        scaler.scale(loss).backward()

        if (i + 1) % iters_to_accumulate == 0:
            # may unscale_ here if desired (e.g., to allow clipping unscaled gradients)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
```

## 4. Gradient Penalty

A gradient penalty implementation commonly creates gradients using ```torch.autograd.grad()```, combines them to create the penalty value, and adds the penalty value to the loss.

Here’s an ordinary example of an L2 penalty without gradient scaling or autocasting:
```python
for epoch in epochs:
    for input, target in data:
        optimizer.zero_grad()
        output = model(input)
        loss = loss_fn(output, target)

        # Creates gradients
        grad_params = torch.autograd.grad(outputs=loss,
                                          inputs=model.parameters(),
                                          create_graph=True)

        # Computes the penalty term and adds it to the loss
        grad_norm = 0
        for grad in grad_params:
            grad_norm += grad.pow(2).sum()
        grad_norm = grad_norm.sqrt()
        loss = loss + grad_norm

        loss.backward()

        # clip gradients here, if desired

        optimizer.step()
```
To implement a gradient penalty with gradient scaling, the ```outputs``` Tensor(s) passed to ```torch.autograd.grad()``` should be scaled. The resulting gradients will therefore be scaled, and should be unscaled before being combined to create the penalty value.

Also, the penalty term computation is part of the forward pass, and therefore should be inside an autocast context.

Here’s how that looks for the same L2 penalty:
```python
scaler = GradScaler()

for epoch in epochs:
    for input, target in data:
        optimizer.zero_grad()
        with autocast():
            output = model(input)
            loss = loss_fn(output, target)

        # Scales the loss for autograd.grad's backward pass, producing scaled_grad_params
        scaled_grad_params = torch.autograd.grad(outputs=scaler.scale(loss),
                                                 inputs=model.parameters(),
                                                 create_graph=True)

        # Creates unscaled grad_params before computing the penalty. scaled_grad_params are
        # not owned by any optimizer, so ordinary division is used instead of scaler.unscale_:
        inv_scale = 1./scaler.get_scale()
        grad_params = [p * inv_scale for p in scaled_grad_params]

        # Computes the penalty term and adds it to the loss
        with autocast():
            grad_norm = 0
            for grad in grad_params:
                grad_norm += grad.pow(2).sum()
            grad_norm = grad_norm.sqrt()
            loss = loss + grad_norm

        # Applies scaling to the backward call as usual.
        # Accumulates leaf gradients that are correctly scaled.
        scaler.scale(loss).backward()

        # may unscale_ here if desired (e.g., to allow clipping unscaled gradients)

        # step() and update() proceed as usual.
        scaler.step(optimizer)
        scaler.update()
```

## 5. Working with Multiple Models, Losses and Optimizers
If your network has multiple losses, you must call ```scaler.scale``` on each of them individually. If your network has multiple optimizers, you may call ```scaler.unscale_``` on any of them individually, and you must call ```scaler.step``` on each of them individually.

However, ```scaler.update``` should only be called once, after all optimizers used this iteration have been stepped:
```python
scaler = torch.cuda.amp.GradScaler()

for epoch in epochs:
    for input, target in data:
        optimizer0.zero_grad()
        optimizer1.zero_grad()
        with autocast():
            output0 = model0(input)
            output1 = model1(input)
            loss0 = loss_fn(2 * output0 + 3 * output1, target)
            loss1 = loss_fn(3 * output0 - 5 * output1, target)

        # (retain_graph here is unrelated to amp, it's present because in this
        # example, both backward() calls share some sections of graph.)
        scaler.scale(loss0).backward(retain_graph=True)
        scaler.scale(loss1).backward()

        # You can choose which optimizers receive explicit unscaling, if you
        # want to inspect or modify the gradients of the params they own.
        scaler.unscale_(optimizer0)

        scaler.step(optimizer0)
        scaler.step(optimizer1)

        scaler.update()
```

## 6. AutoCast and Custom Autograd Functions

If the network uses ```custom autograd functions``` (subclasses of ```torch.autograd.Function```), changes are required for autocast compatibility if any function
- takes multiple floating-point Tensor inputs,
- wraps any autocastable op
- require a particular ```dtype```

In all cases, if you’re importing the function and can’t alter its definition, a safe fallback is to disable autocast and force execution in ```float32``` ( or ```dtype```) at any points of use where errors occur:
```python
with autocast():
    ...
    with autocast(enabled=False):
        output = imported_function(input1.float(), input2.float())
```
If you’re the function’s author (or can alter its definition) a better solution is to use the ```torch.cuda.amp.custom_fwd()``` and ```torch.cuda.amp.custom_bwd()``` decorators as shown in the relevant case below.

## 7. Functions with Multiple Inputs or autocastable ops
Apply ```custom_fwd``` and ```custom_bwd``` (with no arguments) to ```forward``` and ```backward``` respectively. These ensure ```forward``` executes with the current autocast state and ```backward``` executes with the same autocast state as ```forward``` (which can prevent type mismatch errors):
```python
class MyMM(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return a.mm(b)
    @staticmethod
    @custom_bwd
    def backward(ctx, grad):
        a, b = ctx.saved_tensors
        return grad.mm(b.t()), a.t().mm(grad)
```

Now ```MyMM``` can be invoked anywhere, without disabling autocast or manually casting inputs:
```python
mymm = MyMM.apply

with autocast():
    output = mymm(input1, input2)
```

## 8. Functions that need a particular ```dtype```
Consider a custom function that requires ```torch.float32``` inputs. Apply ```custom_fwd```(cast_inputs=torch.float32) to ```forward``` and ```custom_bwd``` (with no arguments) to ```backward```. If forward runs in an autocast-enabled region, the decorators cast floating-point CUDA Tensor inputs to ```float32```, and locally disable autocast during ```forward and backward```:

```python
class MyFloat32Func(torch.autograd.Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, input):
        ctx.save_for_backward(input)
        ...
        return fwd_output
    @staticmethod
    @custom_bwd
    def backward(ctx, grad):
        ...
```
Now ```MyFloat32Func``` can be invoked anywhere, without manually disabling autocast or casting inputs:

```python
func = MyFloat32Func.apply

with autocast():
    # func will run in float32, regardless of the surrounding autocast state
    output = func(input)
```