#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"
#include<cmath>

// Signatures for GPU kernels here 


using namespace tensorflow;
#include "FemGrad.h"


REGISTER_OP("FemGrad")
.Input("u : double")
.Output("grad : double")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    
        shape_inference::ShapeHandle u_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &u_shape));

        c->set_output(0, c->Matrix(-1,2));
    return Status::OK();
  });

REGISTER_OP("FemGradGrad")
.Input("grad_grad : double")
.Input("grad : double")
.Input("u : double")
.Output("grad_u : double");

/*-------------------------------------------------------------------------------------*/

class FemGradOp : public OpKernel {
private:
  
public:
  explicit FemGradOp(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(1, context->num_inputs());
    
    
    const Tensor& u = context->input(0);
    
    
    const TensorShape& u_shape = u.shape();
    
    
    DCHECK_EQ(u_shape.dims(), 1);

    // extra check
        
    // create output shape
    
    TensorShape grad_shape({-1,2});
            
    // create output tensor
    
    Tensor* grad = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_shape, &grad));
    
    // get the corresponding Eigen tensors for data access
    
    auto u_tensor = u.flat<double>().data();
    auto grad_tensor = grad->flat<double>().data();   

    // implement your forward function here 

    // TODO:

  }
};
REGISTER_KERNEL_BUILDER(Name("FemGrad").Device(DEVICE_CPU), FemGradOp);



class FemGradGradOp : public OpKernel {
private:
  
public:
  explicit FemGradGradOp(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_grad = context->input(0);
    const Tensor& grad = context->input(1);
    const Tensor& u = context->input(2);
    
    
    const TensorShape& grad_grad_shape = grad_grad.shape();
    const TensorShape& grad_shape = grad.shape();
    const TensorShape& u_shape = u.shape();
    
    
    DCHECK_EQ(grad_grad_shape.dims(), 2);
    DCHECK_EQ(grad_shape.dims(), 2);
    DCHECK_EQ(u_shape.dims(), 1);

    // extra check
    // int m = Example.dim_size(0);
        
    // create output shape
    
    TensorShape grad_u_shape(u_shape);
            
    // create output tensor
    
    Tensor* grad_u = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_u_shape, &grad_u));
    
    // get the corresponding Eigen tensors for data access
    
    auto u_tensor = u.flat<double>().data();
    auto grad_grad_tensor = grad_grad.flat<double>().data();
    auto grad_tensor = grad.flat<double>().data();
    auto grad_u_tensor = grad_u->flat<double>().data();   

    // implement your backward function here 

    // TODO:
    
  }
};
REGISTER_KERNEL_BUILDER(Name("FemGradGrad").Device(DEVICE_CPU), FemGradGradOp);


/***************************************************************************************
**********************            GPU Operator            ******************************
***************************************************************************************/


#ifdef GOOGLE_CUDA
class FemGradOpGPU : public OpKernel {
private:
  
public:
  explicit FemGradOpGPU(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(1, context->num_inputs());
    
    
    const Tensor& u = context->input(0);
    
    
    const TensorShape& u_shape = u.shape();
    
    
    DCHECK_EQ(u_shape.dims(), 1);

    // extra check
        
    // create output shape
    
    TensorShape grad_shape({-1,2});
            
    // create output tensor
    
    Tensor* grad = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_shape, &grad));
    
    // get the corresponding Eigen tensors for data access
    
    auto u_tensor = u.flat<double>().data();
    auto grad_tensor = grad->flat<double>().data();   

    // implement your forward function here 

    // TODO:

  }
};
REGISTER_KERNEL_BUILDER(Name("FemGrad").Device(DEVICE_GPU), FemGradOpGPU);

class FemGradGradOpGPU : public OpKernel {
private:
  
public:
  explicit FemGradGradOpGPU(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_grad = context->input(0);
    const Tensor& grad = context->input(1);
    const Tensor& u = context->input(2);
    
    
    const TensorShape& grad_grad_shape = grad_grad.shape();
    const TensorShape& grad_shape = grad.shape();
    const TensorShape& u_shape = u.shape();
    
    
    DCHECK_EQ(grad_grad_shape.dims(), 2);
    DCHECK_EQ(grad_shape.dims(), 2);
    DCHECK_EQ(u_shape.dims(), 1);

    // extra check
    // int m = Example.dim_size(0);
        
    // create output shape
    
    TensorShape grad_u_shape(u_shape);
            
    // create output tensor
    
    Tensor* grad_u = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_u_shape, &grad_u));
    
    // get the corresponding Eigen tensors for data access
    
    auto u_tensor = u.flat<double>().data();
    auto grad_grad_tensor = grad_grad.flat<double>().data();
    auto grad_tensor = grad.flat<double>().data();
    auto grad_u_tensor = grad_u->flat<double>().data();   

    // implement your backward function here 

    // TODO:
    
  }
};
REGISTER_KERNEL_BUILDER(Name("FemGradGrad").Device(DEVICE_GPU), FemGradGradOpGPU);

#endif