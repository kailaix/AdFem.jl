#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"
#include<cmath>

// Signatures for GPU kernels here 


using namespace tensorflow;
#include "FemToGaussPointsMfem.h"


REGISTER_OP("FemToGaussPointsMfem")
.Input("u : double")
.Output("out : double")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    
        shape_inference::ShapeHandle u_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &u_shape));

        c->set_output(0, c->Vector(-1));
    return Status::OK();
  });

REGISTER_OP("FemToGaussPointsMfemGrad")
.Input("grad_out : double")
.Input("out : double")
.Input("u : double")
.Output("grad_u : double");

/*-------------------------------------------------------------------------------------*/

class FemToGaussPointsMfemOp : public OpKernel {
private:
  
public:
  explicit FemToGaussPointsMfemOp(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(1, context->num_inputs());
    
    
    const Tensor& u = context->input(0);
    
    
    const TensorShape& u_shape = u.shape();
    
    
    DCHECK_EQ(u_shape.dims(), 1);

    // extra check
        
    // create output shape
    
    TensorShape out_shape({mmesh.ngauss});
            
    // create output tensor
    
    Tensor* out = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &out));
    
    // get the corresponding Eigen tensors for data access
    
    auto u_tensor = u.flat<double>().data();
    auto out_tensor = out->flat<double>().data();   

    // implement your forward function here 

    // TODO:
    MFEM::FemToGaussPointsMfem_forward(out_tensor, u_tensor);

  }
};
REGISTER_KERNEL_BUILDER(Name("FemToGaussPointsMfem").Device(DEVICE_CPU), FemToGaussPointsMfemOp);



class FemToGaussPointsMfemGradOp : public OpKernel {
private:
  
public:
  explicit FemToGaussPointsMfemGradOp(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_out = context->input(0);
    const Tensor& out = context->input(1);
    const Tensor& u = context->input(2);
    
    
    const TensorShape& grad_out_shape = grad_out.shape();
    const TensorShape& out_shape = out.shape();
    const TensorShape& u_shape = u.shape();
    
    
    DCHECK_EQ(grad_out_shape.dims(), 1);
    DCHECK_EQ(out_shape.dims(), 1);
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
    auto grad_out_tensor = grad_out.flat<double>().data();
    auto out_tensor = out.flat<double>().data();
    auto grad_u_tensor = grad_u->flat<double>().data();   

    // implement your backward function here 

    // TODO:
    grad_u->flat<double>().setZero();
    MFEM::FemToGaussPointsMfem_backward(
      grad_u_tensor, grad_out_tensor, out_tensor, u_tensor);
    
  }
};
REGISTER_KERNEL_BUILDER(Name("FemToGaussPointsMfemGrad").Device(DEVICE_CPU), FemToGaussPointsMfemGradOp);
