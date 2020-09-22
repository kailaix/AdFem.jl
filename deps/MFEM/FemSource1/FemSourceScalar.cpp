#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"
#include<cmath>

// Signatures for GPU kernels here 


using namespace tensorflow;
#include "FemSourceScalar.h"


REGISTER_OP("FemSourceScalar")
.Input("f : double")
.Output("rhs : double")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    
        shape_inference::ShapeHandle f_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &f_shape));

        c->set_output(0, c->Vector(-1));
    return Status::OK();
  });

REGISTER_OP("FemSourceScalarGrad")
.Input("grad_rhs : double")
.Input("rhs : double")
.Input("f : double")
.Output("grad_f : double");

/*-------------------------------------------------------------------------------------*/

class FemSourceScalarOp : public OpKernel {
private:
  
public:
  explicit FemSourceScalarOp(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(1, context->num_inputs());
    
    
    const Tensor& f = context->input(0);
    
    
    const TensorShape& f_shape = f.shape();
    
    
    DCHECK_EQ(f_shape.dims(), 1);

    // extra check
        
    // create output shape
    
    TensorShape rhs_shape({mmesh.ndof});
            
    // create output tensor
    
    Tensor* rhs = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, rhs_shape, &rhs));
    
    // get the corresponding Eigen tensors for data access
    
    auto f_tensor = f.flat<double>().data();
    auto rhs_tensor = rhs->flat<double>().data();   

    // implement your forward function here 

    // TODO:
    rhs->flat<double>().setZero();
    MFEM::FemSourceScalar_forward(rhs_tensor, f_tensor);

  }
};
REGISTER_KERNEL_BUILDER(Name("FemSourceScalar").Device(DEVICE_CPU), FemSourceScalarOp);



class FemSourceScalarGradOp : public OpKernel {
private:
  
public:
  explicit FemSourceScalarGradOp(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_rhs = context->input(0);
    const Tensor& rhs = context->input(1);
    const Tensor& f = context->input(2);
    
    
    const TensorShape& grad_rhs_shape = grad_rhs.shape();
    const TensorShape& rhs_shape = rhs.shape();
    const TensorShape& f_shape = f.shape();
    
    
    DCHECK_EQ(grad_rhs_shape.dims(), 1);
    DCHECK_EQ(rhs_shape.dims(), 1);
    DCHECK_EQ(f_shape.dims(), 1);

    // extra check
    // int m = Example.dim_size(0);
        
    // create output shape
    
    TensorShape grad_f_shape(f_shape);
            
    // create output tensor
    
    Tensor* grad_f = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_f_shape, &grad_f));
    
    // get the corresponding Eigen tensors for data access
    
    auto f_tensor = f.flat<double>().data();
    auto grad_rhs_tensor = grad_rhs.flat<double>().data();
    auto rhs_tensor = rhs.flat<double>().data();
    auto grad_f_tensor = grad_f->flat<double>().data();   

    // implement your backward function here 

    // TODO:
    MFEM::FemSourceScalar_backward(grad_f_tensor, grad_rhs_tensor, rhs_tensor, f_tensor);
    
  }
};
REGISTER_KERNEL_BUILDER(Name("FemSourceScalarGrad").Device(DEVICE_CPU), FemSourceScalarGradOp);


/***************************************************************************************
**********************            GPU Operator            ******************************
***************************************************************************************/


#ifdef GOOGLE_CUDA
class FemSourceScalarOpGPU : public OpKernel {
private:
  
public:
  explicit FemSourceScalarOpGPU(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(1, context->num_inputs());
    
    
    const Tensor& f = context->input(0);
    
    
    const TensorShape& f_shape = f.shape();
    
    
    DCHECK_EQ(f_shape.dims(), 1);

    // extra check
        
    // create output shape
    
    TensorShape rhs_shape({-1});
            
    // create output tensor
    
    Tensor* rhs = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, rhs_shape, &rhs));
    
    // get the corresponding Eigen tensors for data access
    
    auto f_tensor = f.flat<double>().data();
    auto rhs_tensor = rhs->flat<double>().data();   

    // implement your forward function here 

    // TODO:

  }
};
REGISTER_KERNEL_BUILDER(Name("FemSourceScalar").Device(DEVICE_GPU), FemSourceScalarOpGPU);

class FemSourceScalarGradOpGPU : public OpKernel {
private:
  
public:
  explicit FemSourceScalarGradOpGPU(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_rhs = context->input(0);
    const Tensor& rhs = context->input(1);
    const Tensor& f = context->input(2);
    
    
    const TensorShape& grad_rhs_shape = grad_rhs.shape();
    const TensorShape& rhs_shape = rhs.shape();
    const TensorShape& f_shape = f.shape();
    
    
    DCHECK_EQ(grad_rhs_shape.dims(), 1);
    DCHECK_EQ(rhs_shape.dims(), 1);
    DCHECK_EQ(f_shape.dims(), 1);

    // extra check
    // int m = Example.dim_size(0);
        
    // create output shape
    
    TensorShape grad_f_shape(f_shape);
            
    // create output tensor
    
    Tensor* grad_f = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_f_shape, &grad_f));
    
    // get the corresponding Eigen tensors for data access
    
    auto f_tensor = f.flat<double>().data();
    auto grad_rhs_tensor = grad_rhs.flat<double>().data();
    auto rhs_tensor = rhs.flat<double>().data();
    auto grad_f_tensor = grad_f->flat<double>().data();   

    // implement your backward function here 

    // TODO:
    
  }
};
REGISTER_KERNEL_BUILDER(Name("FemSourceScalarGrad").Device(DEVICE_GPU), FemSourceScalarGradOpGPU);

#endif