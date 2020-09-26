#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"
#include<cmath>

// Signatures for GPU kernels here 


using namespace tensorflow;
#include "PlaneStrainAndStress.h"


REGISTER_OP("PlaneStrainAndStress")
.Input("e : double")
.Input("nu : double")
.Input("mode : int32")
.Output("hmat : double")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    
        shape_inference::ShapeHandle e_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &e_shape));
        shape_inference::ShapeHandle nu_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &nu_shape));
        shape_inference::ShapeHandle mode_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &mode_shape));

        c->set_output(0, c->MakeShape({-1,3,3}));
    return Status::OK();
  });

REGISTER_OP("PlaneStrainAndStressGrad")
.Input("grad_hmat : double")
.Input("hmat : double")
.Input("e : double")
.Input("nu : double")
.Input("mode : int32")
.Output("grad_e : double")
.Output("grad_nu : double")
.Output("grad_mode : int32");

/*-------------------------------------------------------------------------------------*/

class PlaneStrainAndStressOp : public OpKernel {
private:
  
public:
  explicit PlaneStrainAndStressOp(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(3, context->num_inputs());
    
    
    const Tensor& e = context->input(0);
    const Tensor& nu = context->input(1);
    const Tensor& mode = context->input(2);
    
    
    const TensorShape& e_shape = e.shape();
    const TensorShape& nu_shape = nu.shape();
    const TensorShape& mode_shape = mode.shape();
    
    
    DCHECK_EQ(e_shape.dims(), 1);
    DCHECK_EQ(nu_shape.dims(), 1);
    DCHECK_EQ(mode_shape.dims(), 0);

    // extra check
        
    // create output shape
    int N = e_shape.dim_size(0);
    TensorShape hmat_shape({N,3,3});
            
    // create output tensor
    
    Tensor* hmat = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, hmat_shape, &hmat));
    
    // get the corresponding Eigen tensors for data access
    
    auto e_tensor = e.flat<double>().data();
    auto nu_tensor = nu.flat<double>().data();
    auto mode_tensor = mode.flat<int32>().data();
    auto hmat_tensor = hmat->flat<double>().data();   

    // implement your forward function here 

    // TODO:
    if (*mode_tensor == 0)
      PlaneStrainMatrix_forward(hmat_tensor, e_tensor, nu_tensor, N);
    else if (*mode_tensor == 1)
      PlaneStressMatrix_forward(hmat_tensor, e_tensor, nu_tensor, N);

  }
};
REGISTER_KERNEL_BUILDER(Name("PlaneStrainAndStress").Device(DEVICE_CPU), PlaneStrainAndStressOp);



class PlaneStrainAndStressGradOp : public OpKernel {
private:
  
public:
  explicit PlaneStrainAndStressGradOp(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_hmat = context->input(0);
    const Tensor& hmat = context->input(1);
    const Tensor& e = context->input(2);
    const Tensor& nu = context->input(3);
    const Tensor& mode = context->input(4);
    
    
    const TensorShape& grad_hmat_shape = grad_hmat.shape();
    const TensorShape& hmat_shape = hmat.shape();
    const TensorShape& e_shape = e.shape();
    const TensorShape& nu_shape = nu.shape();
    const TensorShape& mode_shape = mode.shape();
    
    
    DCHECK_EQ(grad_hmat_shape.dims(), 3);
    DCHECK_EQ(hmat_shape.dims(), 3);
    DCHECK_EQ(e_shape.dims(), 1);
    DCHECK_EQ(nu_shape.dims(), 1);
    DCHECK_EQ(mode_shape.dims(), 0);

    // extra check
    // int m = Example.dim_size(0);
        
    // create output shape
    
    TensorShape grad_e_shape(e_shape);
    TensorShape grad_nu_shape(nu_shape);
    TensorShape grad_mode_shape(mode_shape);
            
    // create output tensor
    
    Tensor* grad_e = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_e_shape, &grad_e));
    Tensor* grad_nu = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, grad_nu_shape, &grad_nu));
    Tensor* grad_mode = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(2, grad_mode_shape, &grad_mode));
    
    // get the corresponding Eigen tensors for data access
    
    auto e_tensor = e.flat<double>().data();
    auto nu_tensor = nu.flat<double>().data();
    auto mode_tensor = mode.flat<int32>().data();
    auto grad_hmat_tensor = grad_hmat.flat<double>().data();
    auto hmat_tensor = hmat.flat<double>().data();
    auto grad_e_tensor = grad_e->flat<double>().data();
    auto grad_nu_tensor = grad_nu->flat<double>().data();   

    // implement your backward function here 

    // TODO:
    int N = e_shape.dim_size(0);
    if (*mode_tensor == 0)
      PlaneStrainMatrix_backward(grad_nu_tensor, grad_e_tensor, grad_hmat_tensor,  e_tensor, nu_tensor, N);
    else if (*mode_tensor == 1)
      PlaneStressMatrix_backward(grad_nu_tensor, grad_e_tensor, grad_hmat_tensor,  e_tensor, nu_tensor, N);
    
  }
};
REGISTER_KERNEL_BUILDER(Name("PlaneStrainAndStressGrad").Device(DEVICE_CPU), PlaneStrainAndStressGradOp);


/***************************************************************************************
**********************            GPU Operator            ******************************
***************************************************************************************/


#ifdef GOOGLE_CUDA
class PlaneStrainAndStressOpGPU : public OpKernel {
private:
  
public:
  explicit PlaneStrainAndStressOpGPU(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(3, context->num_inputs());
    
    
    const Tensor& e = context->input(0);
    const Tensor& nu = context->input(1);
    const Tensor& mode = context->input(2);
    
    
    const TensorShape& e_shape = e.shape();
    const TensorShape& nu_shape = nu.shape();
    const TensorShape& mode_shape = mode.shape();
    
    
    DCHECK_EQ(e_shape.dims(), 1);
    DCHECK_EQ(nu_shape.dims(), 1);
    DCHECK_EQ(mode_shape.dims(), 0);

    // extra check
        
    // create output shape
    
    TensorShape hmat_shape({-1,3,3});
            
    // create output tensor
    
    Tensor* hmat = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, hmat_shape, &hmat));
    
    // get the corresponding Eigen tensors for data access
    
    auto e_tensor = e.flat<double>().data();
    auto nu_tensor = nu.flat<double>().data();
    auto mode_tensor = mode.flat<int32>().data();
    auto hmat_tensor = hmat->flat<double>().data();   

    // implement your forward function here 

    // TODO:

  }
};
REGISTER_KERNEL_BUILDER(Name("PlaneStrainAndStress").Device(DEVICE_GPU), PlaneStrainAndStressOpGPU);

class PlaneStrainAndStressGradOpGPU : public OpKernel {
private:
  
public:
  explicit PlaneStrainAndStressGradOpGPU(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_hmat = context->input(0);
    const Tensor& hmat = context->input(1);
    const Tensor& e = context->input(2);
    const Tensor& nu = context->input(3);
    const Tensor& mode = context->input(4);
    
    
    const TensorShape& grad_hmat_shape = grad_hmat.shape();
    const TensorShape& hmat_shape = hmat.shape();
    const TensorShape& e_shape = e.shape();
    const TensorShape& nu_shape = nu.shape();
    const TensorShape& mode_shape = mode.shape();
    
    
    DCHECK_EQ(grad_hmat_shape.dims(), 3);
    DCHECK_EQ(hmat_shape.dims(), 3);
    DCHECK_EQ(e_shape.dims(), 1);
    DCHECK_EQ(nu_shape.dims(), 1);
    DCHECK_EQ(mode_shape.dims(), 0);

    // extra check
    // int m = Example.dim_size(0);
        
    // create output shape
    
    TensorShape grad_e_shape(e_shape);
    TensorShape grad_nu_shape(nu_shape);
    TensorShape grad_mode_shape(mode_shape);
            
    // create output tensor
    
    Tensor* grad_e = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_e_shape, &grad_e));
    Tensor* grad_nu = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, grad_nu_shape, &grad_nu));
    Tensor* grad_mode = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(2, grad_mode_shape, &grad_mode));
    
    // get the corresponding Eigen tensors for data access
    
    auto e_tensor = e.flat<double>().data();
    auto nu_tensor = nu.flat<double>().data();
    auto mode_tensor = mode.flat<int32>().data();
    auto grad_hmat_tensor = grad_hmat.flat<double>().data();
    auto hmat_tensor = hmat.flat<double>().data();
    auto grad_e_tensor = grad_e->flat<double>().data();
    auto grad_nu_tensor = grad_nu->flat<double>().data();   

    // implement your backward function here 

    // TODO:
    
  }
};
REGISTER_KERNEL_BUILDER(Name("PlaneStrainAndStressGrad").Device(DEVICE_GPU), PlaneStrainAndStressGradOpGPU);

#endif