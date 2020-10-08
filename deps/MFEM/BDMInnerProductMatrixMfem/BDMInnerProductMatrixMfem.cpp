#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"
#include<cmath>

// Signatures for GPU kernels here 


using namespace tensorflow;
#include "BDMInnerProductMatrixMfem.h"


REGISTER_OP("BDMInnerProductMatrixMfem")
.Input("alpha : double")
.Input("beta : double")
.Output("indices : int64")
.Output("v : double")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    
        shape_inference::ShapeHandle alpha_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &alpha_shape));
        shape_inference::ShapeHandle beta_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &beta_shape));

        c->set_output(0, c->Matrix(-1,2));
        c->set_output(1, c->Vector(-1));
    return Status::OK();
  });

REGISTER_OP("BDMInnerProductMatrixMfemGrad")
.Input("grad_v : double")
.Input("indices : int64")
.Input("v : double")
.Input("alpha : double")
.Input("beta : double")
.Output("grad_alpha : double")
.Output("grad_beta : double");

/*-------------------------------------------------------------------------------------*/

class BDMInnerProductMatrixMfemOp : public OpKernel {
private:
  
public:
  explicit BDMInnerProductMatrixMfemOp(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(2, context->num_inputs());
    
    
    const Tensor& alpha = context->input(0);
    const Tensor& beta = context->input(1);
    
    
    const TensorShape& alpha_shape = alpha.shape();
    const TensorShape& beta_shape = beta.shape();
    
    
    DCHECK_EQ(alpha_shape.dims(), 1);
    DCHECK_EQ(beta_shape.dims(), 1);

    // extra check
        
    // create output shape
    int N = mmesh.elem_ndof * mmesh.ngauss * 6;
    
    TensorShape indices_shape({N,2});
    TensorShape v_shape({N});
            
    // create output tensor
    
    Tensor* indices = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, indices_shape, &indices));
    Tensor* v = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, v_shape, &v));
    
    // get the corresponding Eigen tensors for data access
    
    auto alpha_tensor = alpha.flat<double>().data();
    auto beta_tensor = beta.flat<double>().data();
    auto indices_tensor = indices->flat<int64>().data();
    auto v_tensor = v->flat<double>().data();   

    // implement your forward function here 

    // TODO:
    MFEM::BDMInnerProductMatrixMfem_forward(indices_tensor, v_tensor, alpha_tensor, beta_tensor);

  }
};
REGISTER_KERNEL_BUILDER(Name("BDMInnerProductMatrixMfem").Device(DEVICE_CPU), BDMInnerProductMatrixMfemOp);



class BDMInnerProductMatrixMfemGradOp : public OpKernel {
private:
  
public:
  explicit BDMInnerProductMatrixMfemGradOp(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_v = context->input(0);
    const Tensor& indices = context->input(1);
    const Tensor& v = context->input(2);
    const Tensor& alpha = context->input(3);
    const Tensor& beta = context->input(4);
    
    
    const TensorShape& grad_v_shape = grad_v.shape();
    const TensorShape& indices_shape = indices.shape();
    const TensorShape& v_shape = v.shape();
    const TensorShape& alpha_shape = alpha.shape();
    const TensorShape& beta_shape = beta.shape();
    
    
    DCHECK_EQ(grad_v_shape.dims(), 1);
    DCHECK_EQ(indices_shape.dims(), 2);
    DCHECK_EQ(v_shape.dims(), 1);
    DCHECK_EQ(alpha_shape.dims(), 1);
    DCHECK_EQ(beta_shape.dims(), 1);

    // extra check
    // int m = Example.dim_size(0);
        
    // create output shape
    
    TensorShape grad_alpha_shape(alpha_shape);
    TensorShape grad_beta_shape(beta_shape);
            
    // create output tensor
    
    Tensor* grad_alpha = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_alpha_shape, &grad_alpha));
    Tensor* grad_beta = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, grad_beta_shape, &grad_beta));
    
    // get the corresponding Eigen tensors for data access
    
    auto alpha_tensor = alpha.flat<double>().data();
    auto beta_tensor = beta.flat<double>().data();
    auto grad_v_tensor = grad_v.flat<double>().data();
    auto indices_tensor = indices.flat<int64>().data();
    auto v_tensor = v.flat<double>().data();
    auto grad_alpha_tensor = grad_alpha->flat<double>().data();
    auto grad_beta_tensor = grad_beta->flat<double>().data();   

    // implement your backward function here 

    // TODO:
    
  }
};
REGISTER_KERNEL_BUILDER(Name("BDMInnerProductMatrixMfemGrad").Device(DEVICE_CPU), BDMInnerProductMatrixMfemGradOp);


/***************************************************************************************
**********************            GPU Operator            ******************************
***************************************************************************************/


#ifdef GOOGLE_CUDA
class BDMInnerProductMatrixMfemOpGPU : public OpKernel {
private:
  
public:
  explicit BDMInnerProductMatrixMfemOpGPU(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(2, context->num_inputs());
    
    
    const Tensor& alpha = context->input(0);
    const Tensor& beta = context->input(1);
    
    
    const TensorShape& alpha_shape = alpha.shape();
    const TensorShape& beta_shape = beta.shape();
    
    
    DCHECK_EQ(alpha_shape.dims(), 1);
    DCHECK_EQ(beta_shape.dims(), 1);

    // extra check
        
    // create output shape
    
    TensorShape indices_shape({-1,2});
    TensorShape v_shape({-1});
            
    // create output tensor
    
    Tensor* indices = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, indices_shape, &indices));
    Tensor* v = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, v_shape, &v));
    
    // get the corresponding Eigen tensors for data access
    
    auto alpha_tensor = alpha.flat<double>().data();
    auto beta_tensor = beta.flat<double>().data();
    auto indices_tensor = indices->flat<int64>().data();
    auto v_tensor = v->flat<double>().data();   

    // implement your forward function here 

    // TODO:

  }
};
REGISTER_KERNEL_BUILDER(Name("BDMInnerProductMatrixMfem").Device(DEVICE_GPU), BDMInnerProductMatrixMfemOpGPU);

class BDMInnerProductMatrixMfemGradOpGPU : public OpKernel {
private:
  
public:
  explicit BDMInnerProductMatrixMfemGradOpGPU(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_v = context->input(0);
    const Tensor& indices = context->input(1);
    const Tensor& v = context->input(2);
    const Tensor& alpha = context->input(3);
    const Tensor& beta = context->input(4);
    
    
    const TensorShape& grad_v_shape = grad_v.shape();
    const TensorShape& indices_shape = indices.shape();
    const TensorShape& v_shape = v.shape();
    const TensorShape& alpha_shape = alpha.shape();
    const TensorShape& beta_shape = beta.shape();
    
    
    DCHECK_EQ(grad_v_shape.dims(), 1);
    DCHECK_EQ(indices_shape.dims(), 2);
    DCHECK_EQ(v_shape.dims(), 1);
    DCHECK_EQ(alpha_shape.dims(), 1);
    DCHECK_EQ(beta_shape.dims(), 1);

    // extra check
    // int m = Example.dim_size(0);
        
    // create output shape
    
    TensorShape grad_alpha_shape(alpha_shape);
    TensorShape grad_beta_shape(beta_shape);
            
    // create output tensor
    
    Tensor* grad_alpha = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_alpha_shape, &grad_alpha));
    Tensor* grad_beta = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, grad_beta_shape, &grad_beta));
    
    // get the corresponding Eigen tensors for data access
    
    auto alpha_tensor = alpha.flat<double>().data();
    auto beta_tensor = beta.flat<double>().data();
    auto grad_v_tensor = grad_v.flat<double>().data();
    auto indices_tensor = indices.flat<int64>().data();
    auto v_tensor = v.flat<double>().data();
    auto grad_alpha_tensor = grad_alpha->flat<double>().data();
    auto grad_beta_tensor = grad_beta->flat<double>().data();   

    // implement your backward function here 

    // TODO:
    
  }
};
REGISTER_KERNEL_BUILDER(Name("BDMInnerProductMatrixMfemGrad").Device(DEVICE_GPU), BDMInnerProductMatrixMfemGradOpGPU);

#endif