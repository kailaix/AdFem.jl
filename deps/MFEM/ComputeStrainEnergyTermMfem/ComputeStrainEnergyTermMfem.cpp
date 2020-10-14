#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"
#include<cmath>

// Signatures for GPU kernels here 


using namespace tensorflow;
#include "ComputeStrainEnergyTermMfem.h"


REGISTER_OP("ComputeStrainEnergyTermMfem")
.Input("sigma : double")
.Output("out : double")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    
        shape_inference::ShapeHandle sigma_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &sigma_shape));

        c->set_output(0, c->Vector(-1));
    return Status::OK();
  });

REGISTER_OP("ComputeStrainEnergyTermMfemGrad")
.Input("grad_out : double")
.Input("out : double")
.Input("sigma : double")
.Output("grad_sigma : double");

/*-------------------------------------------------------------------------------------*/

class ComputeStrainEnergyTermMfemOp : public OpKernel {
private:
  
public:
  explicit ComputeStrainEnergyTermMfemOp(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(1, context->num_inputs());
    
    
    const Tensor& sigma = context->input(0);
    
    
    const TensorShape& sigma_shape = sigma.shape();
    
    
    DCHECK_EQ(sigma_shape.dims(), 2);

    // extra check
        
    // create output shape
    
    TensorShape out_shape({2*mmesh.ndof});
            
    // create output tensor
    
    Tensor* out = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &out));
    
    // get the corresponding Eigen tensors for data access
    
    auto sigma_tensor = sigma.flat<double>().data();
    auto out_tensor = out->flat<double>().data();   

    // implement your forward function here 

    // TODO:
    out->flat<double>().setZero();
    MFEM::ComputeStrainEnergyTermMfem_forward(out_tensor, sigma_tensor);

  }
};
REGISTER_KERNEL_BUILDER(Name("ComputeStrainEnergyTermMfem").Device(DEVICE_CPU), ComputeStrainEnergyTermMfemOp);



class ComputeStrainEnergyTermMfemGradOp : public OpKernel {
private:
  
public:
  explicit ComputeStrainEnergyTermMfemGradOp(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_out = context->input(0);
    const Tensor& out = context->input(1);
    const Tensor& sigma = context->input(2);
    
    
    const TensorShape& grad_out_shape = grad_out.shape();
    const TensorShape& out_shape = out.shape();
    const TensorShape& sigma_shape = sigma.shape();
    
    
    DCHECK_EQ(grad_out_shape.dims(), 1);
    DCHECK_EQ(out_shape.dims(), 1);
    DCHECK_EQ(sigma_shape.dims(), 2);

    // extra check
    // int m = Example.dim_size(0);
        
    // create output shape
    
    TensorShape grad_sigma_shape(sigma_shape);
            
    // create output tensor
    
    Tensor* grad_sigma = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_sigma_shape, &grad_sigma));
    
    // get the corresponding Eigen tensors for data access
    
    auto sigma_tensor = sigma.flat<double>().data();
    auto grad_out_tensor = grad_out.flat<double>().data();
    auto out_tensor = out.flat<double>().data();
    auto grad_sigma_tensor = grad_sigma->flat<double>().data();   

    // implement your backward function here 

    // TODO:
    grad_sigma->flat<double>().setZero();
    MFEM::ComputeStrainEnergyTermMfem_backward(grad_sigma_tensor, grad_out_tensor);
    
  }
};
REGISTER_KERNEL_BUILDER(Name("ComputeStrainEnergyTermMfemGrad").Device(DEVICE_CPU), ComputeStrainEnergyTermMfemGradOp);

