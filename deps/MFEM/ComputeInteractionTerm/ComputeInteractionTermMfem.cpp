#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"
#include<cmath>

// Signatures for GPU kernels here 


using namespace tensorflow;
#include "ComputeInteractionTermMfem.h"


REGISTER_OP("ComputeInteractionTermMfem")
.Input("p : double")
.Output("out : double")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    
        shape_inference::ShapeHandle p_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &p_shape));

        c->set_output(0, c->Vector(-1));
    return Status::OK();
  });

REGISTER_OP("ComputeInteractionTermMfemGrad")
.Input("grad_out : double")
.Input("out : double")
.Input("p : double")
.Output("grad_p : double");

/*-------------------------------------------------------------------------------------*/

class ComputeInteractionTermMfemOp : public OpKernel {
private:
  
public:
  explicit ComputeInteractionTermMfemOp(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(1, context->num_inputs());
    
    
    const Tensor& p = context->input(0);
    
    
    const TensorShape& p_shape = p.shape();
    
    
    DCHECK_EQ(p_shape.dims(), 1);

    // extra check
        
    // create output shape
    
    TensorShape out_shape({2*mmesh.ndof});
            
    // create output tensor
    
    Tensor* out = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &out));
    
    // get the corresponding Eigen tensors for data access
    
    auto p_tensor = p.flat<double>().data();
    auto out_tensor = out->flat<double>().data();   

    // implement your forward function here 

    // TODO:
    out->flat<double>().setZero();
    MFEM::ComputeInteractionTermMfem_forward(out_tensor, p_tensor);

  }
};
REGISTER_KERNEL_BUILDER(Name("ComputeInteractionTermMfem").Device(DEVICE_CPU), ComputeInteractionTermMfemOp);



class ComputeInteractionTermMfemGradOp : public OpKernel {
private:
  
public:
  explicit ComputeInteractionTermMfemGradOp(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_out = context->input(0);
    const Tensor& out = context->input(1);
    const Tensor& p = context->input(2);
    
    
    const TensorShape& grad_out_shape = grad_out.shape();
    const TensorShape& out_shape = out.shape();
    const TensorShape& p_shape = p.shape();
    
    
    DCHECK_EQ(grad_out_shape.dims(), 1);
    DCHECK_EQ(out_shape.dims(), 1);
    DCHECK_EQ(p_shape.dims(), 1);

    // extra check
    // int m = Example.dim_size(0);
        
    // create output shape
    
    TensorShape grad_p_shape(p_shape);
            
    // create output tensor
    
    Tensor* grad_p = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_p_shape, &grad_p));
    
    // get the corresponding Eigen tensors for data access
    
    auto p_tensor = p.flat<double>().data();
    auto grad_out_tensor = grad_out.flat<double>().data();
    auto out_tensor = out.flat<double>().data();
    auto grad_p_tensor = grad_p->flat<double>().data();   

    // implement your backward function here 

    // TODO:
    grad_p->flat<double>().setZero();
    MFEM::ComputeInteractionTermMfem_backward(grad_p_tensor, grad_out_tensor, out_tensor, p_tensor);
    
  }
};
REGISTER_KERNEL_BUILDER(Name("ComputeInteractionTermMfemGrad").Device(DEVICE_CPU), ComputeInteractionTermMfemGradOp);
