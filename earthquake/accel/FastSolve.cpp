#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"
#include<cmath>


#ifdef USE_GPU
#include "tensorflow/core/util/gpu_kernel_helper.h"
namespace tensorflow{
  typedef Eigen::GpuDevice GPUDevice;
  void forwardGPU(const GPUDevice &d);
  void backwardGPU(const GPUDevice &d);
}
#endif
using namespace tensorflow;
#include "FastSolve.h"


REGISTER_OP("FastSolve")

.Input("rhs : double")
.Output("u : double")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    
        shape_inference::ShapeHandle rhs_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &rhs_shape));

        c->set_output(0, c->Vector(-1));
    return Status::OK();
  });

REGISTER_OP("FastSolveGrad")

.Input("grad_u : double")
.Input("u : double")
.Input("rhs : double")
.Output("grad_rhs : double");


class FastSolveOp : public OpKernel {
private:
  
public:
  explicit FastSolveOp(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(1, context->num_inputs());
    
    
    const Tensor& rhs = context->input(0);
    
    
    const TensorShape& rhs_shape = rhs.shape();
    
    
    DCHECK_EQ(rhs_shape.dims(), 1);

    // extra check
        
    // create output shape
    int n = rhs_shape.dim_size(0);
    TensorShape u_shape({n});
            
    // create output tensor
    
    Tensor* u = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, u_shape, &u));
    
    // get the corresponding Eigen tensors for data access
    
    auto rhs_tensor = rhs.flat<double>().data();
    auto u_tensor = u->flat<double>().data();   

    // implement your forward function here 

    // TODO:
    forward(u_tensor, rhs_tensor, n);

  }
};
REGISTER_KERNEL_BUILDER(Name("FastSolve").Device(DEVICE_CPU), FastSolveOp);



class FastSolveGradOp : public OpKernel {
private:
  
public:
  explicit FastSolveGradOp(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_u = context->input(0);
    const Tensor& u = context->input(1);
    const Tensor& rhs = context->input(2);
    
    
    const TensorShape& grad_u_shape = grad_u.shape();
    const TensorShape& u_shape = u.shape();
    const TensorShape& rhs_shape = rhs.shape();
    
    
    DCHECK_EQ(grad_u_shape.dims(), 1);
    DCHECK_EQ(u_shape.dims(), 1);
    DCHECK_EQ(rhs_shape.dims(), 1);

    // extra check
    // int m = Example.dim_size(0);
    int n = u_shape.dim_size(0);
    // create output shape
    
    TensorShape grad_rhs_shape(rhs_shape);
            
    // create output tensor
    
    Tensor* grad_rhs = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_rhs_shape, &grad_rhs));
    
    // get the corresponding Eigen tensors for data access
    
    auto rhs_tensor = rhs.flat<double>().data();
    auto grad_u_tensor = grad_u.flat<double>().data();
    auto u_tensor = u.flat<double>().data();
    auto grad_rhs_tensor = grad_rhs->flat<double>().data();   

    // implement your backward function here 

    // TODO:
    backward(grad_rhs_tensor, grad_u_tensor, u_tensor, rhs_tensor, n);

    
  }
};
REGISTER_KERNEL_BUILDER(Name("FastSolveGrad").Device(DEVICE_CPU), FastSolveGradOp);


/***************************************************************************************
**********************            GPU Operator            ******************************
***************************************************************************************/


#ifdef USE_GPU
class FastSolveOpGPU : public OpKernel {
private:
  
public:
  explicit FastSolveOpGPU(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(1, context->num_inputs());
    
    
    const Tensor& rhs = context->input(0);
    
    
    const TensorShape& rhs_shape = rhs.shape();
    
    
    DCHECK_EQ(rhs_shape.dims(), 1);

    // extra check
        
    // create output shape
    
    TensorShape u_shape({-1});
            
    // create output tensor
    
    Tensor* u = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, u_shape, &u));
    
    // get the corresponding Eigen tensors for data access
    
    auto rhs_tensor = rhs.flat<double>().data();
    auto u_tensor = u->flat<double>().data();   

    // implement your forward function here 

    // TODO:

  }
};
REGISTER_KERNEL_BUILDER(Name("FastSolve").Device(DEVICE_GPU), FastSolveOpGPU);

class FastSolveGradOpGPU : public OpKernel {
private:
  
public:
  explicit FastSolveGradOpGPU(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_u = context->input(0);
    const Tensor& u = context->input(1);
    const Tensor& rhs = context->input(2);
    
    
    const TensorShape& grad_u_shape = grad_u.shape();
    const TensorShape& u_shape = u.shape();
    const TensorShape& rhs_shape = rhs.shape();
    
    
    DCHECK_EQ(grad_u_shape.dims(), 1);
    DCHECK_EQ(u_shape.dims(), 1);
    DCHECK_EQ(rhs_shape.dims(), 1);

    // extra check
    // int m = Example.dim_size(0);
        
    // create output shape
    
    TensorShape grad_rhs_shape(rhs_shape);
            
    // create output tensor
    
    Tensor* grad_rhs = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_rhs_shape, &grad_rhs));
    
    // get the corresponding Eigen tensors for data access
    
    auto rhs_tensor = rhs.flat<double>().data();
    auto grad_u_tensor = grad_u.flat<double>().data();
    auto u_tensor = u.flat<double>().data();
    auto grad_rhs_tensor = grad_rhs->flat<double>().data();   

    // implement your backward function here 

    // TODO:
    
  }
};
REGISTER_KERNEL_BUILDER(Name("FastSolveGrad").Device(DEVICE_GPU), FastSolveGradOpGPU);

#endif