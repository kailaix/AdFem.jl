#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"
#include<cmath>

// Signatures for GPU kernels here 


using namespace tensorflow;
#include "EvalStrainOnGaussPts.h"


REGISTER_OP("EvalStrainOnGaussPts")
.Input("u : double")
.Output("varespilon : double")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    
        shape_inference::ShapeHandle u_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &u_shape));

        c->set_output(0, c->Matrix(-1,3));
    return Status::OK();
  });

REGISTER_OP("EvalStrainOnGaussPtsGrad")
.Input("grad_varespilon : double")
.Input("varespilon : double")
.Input("u : double")
.Output("grad_u : double");

/*-------------------------------------------------------------------------------------*/

class EvalStrainOnGaussPtsOp : public OpKernel {
private:
  
public:
  explicit EvalStrainOnGaussPtsOp(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(1, context->num_inputs());
    
    
    const Tensor& u = context->input(0);
    
    
    const TensorShape& u_shape = u.shape();
    
    
    DCHECK_EQ(u_shape.dims(), 1);

    // extra check
        
    // create output shape
    
    TensorShape varespilon_shape({mmesh.ngauss,3});
            
    // create output tensor
    
    Tensor* varespilon = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, varespilon_shape, &varespilon));
    
    // get the corresponding Eigen tensors for data access
    
    auto u_tensor = u.flat<double>().data();
    auto varespilon_tensor = varespilon->flat<double>().data();   

    // implement your forward function here 

    // TODO:
    varespilon->flat<double>().setZero();
    MFEM::EvalStrainOnGaussPts_forward(varespilon_tensor, u_tensor);

  }
};
REGISTER_KERNEL_BUILDER(Name("EvalStrainOnGaussPts").Device(DEVICE_CPU), EvalStrainOnGaussPtsOp);



class EvalStrainOnGaussPtsGradOp : public OpKernel {
private:
  
public:
  explicit EvalStrainOnGaussPtsGradOp(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_varespilon = context->input(0);
    const Tensor& varespilon = context->input(1);
    const Tensor& u = context->input(2);
    
    
    const TensorShape& grad_varespilon_shape = grad_varespilon.shape();
    const TensorShape& varespilon_shape = varespilon.shape();
    const TensorShape& u_shape = u.shape();
    
    
    DCHECK_EQ(grad_varespilon_shape.dims(), 2);
    DCHECK_EQ(varespilon_shape.dims(), 2);
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
    auto grad_varespilon_tensor = grad_varespilon.flat<double>().data();
    auto varespilon_tensor = varespilon.flat<double>().data();
    auto grad_u_tensor = grad_u->flat<double>().data();   

    // implement your backward function here 

    // TODO:
    grad_u->flat<double>().setZero();
    MFEM::EvalStrainOnGaussPts_backward(grad_u_tensor, grad_varespilon_tensor);
    
  }
};
REGISTER_KERNEL_BUILDER(Name("EvalStrainOnGaussPtsGrad").Device(DEVICE_CPU), EvalStrainOnGaussPtsGradOp);


/***************************************************************************************
**********************            GPU Operator            ******************************
***************************************************************************************/


#ifdef GOOGLE_CUDA
class EvalStrainOnGaussPtsOpGPU : public OpKernel {
private:
  
public:
  explicit EvalStrainOnGaussPtsOpGPU(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(1, context->num_inputs());
    
    
    const Tensor& u = context->input(0);
    
    
    const TensorShape& u_shape = u.shape();
    
    
    DCHECK_EQ(u_shape.dims(), 1);

    // extra check
        
    // create output shape
    
    TensorShape varespilon_shape({-1,3});
            
    // create output tensor
    
    Tensor* varespilon = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, varespilon_shape, &varespilon));
    
    // get the corresponding Eigen tensors for data access
    
    auto u_tensor = u.flat<double>().data();
    auto varespilon_tensor = varespilon->flat<double>().data();   

    // implement your forward function here 

    // TODO:

  }
};
REGISTER_KERNEL_BUILDER(Name("EvalStrainOnGaussPts").Device(DEVICE_GPU), EvalStrainOnGaussPtsOpGPU);

class EvalStrainOnGaussPtsGradOpGPU : public OpKernel {
private:
  
public:
  explicit EvalStrainOnGaussPtsGradOpGPU(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_varespilon = context->input(0);
    const Tensor& varespilon = context->input(1);
    const Tensor& u = context->input(2);
    
    
    const TensorShape& grad_varespilon_shape = grad_varespilon.shape();
    const TensorShape& varespilon_shape = varespilon.shape();
    const TensorShape& u_shape = u.shape();
    
    
    DCHECK_EQ(grad_varespilon_shape.dims(), 2);
    DCHECK_EQ(varespilon_shape.dims(), 2);
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
    auto grad_varespilon_tensor = grad_varespilon.flat<double>().data();
    auto varespilon_tensor = varespilon.flat<double>().data();
    auto grad_u_tensor = grad_u->flat<double>().data();   

    // implement your backward function here 

    // TODO:
    
  }
};
REGISTER_KERNEL_BUILDER(Name("EvalStrainOnGaussPtsGrad").Device(DEVICE_GPU), EvalStrainOnGaussPtsGradOpGPU);

#endif