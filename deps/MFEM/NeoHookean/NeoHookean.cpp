#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"
#include<cmath>

// Signatures for GPU kernels here 


using namespace tensorflow;
#include "NeoHookean.h"


REGISTER_OP("NeoHookean")
.Input("u : double")
.Input("mu : double")
.Input("lamb : double")
.Output("psi : double")
.Output("indices : int64")
.Output("vv : double")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    
        shape_inference::ShapeHandle u_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &u_shape));
        shape_inference::ShapeHandle mu_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &mu_shape));
        shape_inference::ShapeHandle lamb_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &lamb_shape));

        c->set_output(0, c->Vector(-1));
        c->set_output(1, c->Matrix(-1, 2));
        c->set_output(2, c->Vector(-1));
    return Status::OK();
  });

REGISTER_OP("NeoHookeanGrad")
.Input("grad_psi : double")
.Input("grad_vv : double")
.Input("psi : double")
.Input("indices : int64")
.Input("vv : double")
.Input("u : double")
.Input("mu : double")
.Input("lamb : double")
.Output("grad_u : double")
.Output("grad_mu : double")
.Output("grad_lamb : double");

/*-------------------------------------------------------------------------------------*/

class NeoHookeanOp : public OpKernel {
private:
  
public:
  explicit NeoHookeanOp(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(3, context->num_inputs());
    
    
    const Tensor& u = context->input(0);
    const Tensor& mu = context->input(1);
    const Tensor& lamb = context->input(2);
    
    
    const TensorShape& u_shape = u.shape();
    const TensorShape& mu_shape = mu.shape();
    const TensorShape& lamb_shape = lamb.shape();
    
    
    DCHECK_EQ(u_shape.dims(), 1);
    DCHECK_EQ(mu_shape.dims(), 1);
    DCHECK_EQ(lamb_shape.dims(), 1);

    // extra check
        
    // create output shape
    
    TensorShape psi_shape({2*mmesh.ndof});
    TensorShape indices_shape({mmesh.ngauss * (2*mmesh.elem_ndof) * (2*mmesh.elem_ndof), 2});
    TensorShape vv_shape({mmesh.ngauss * (2*mmesh.elem_ndof) * (2*mmesh.elem_ndof)});
            
    // create output tensor
    
    Tensor* psi = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, psi_shape, &psi));
    Tensor* indices = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, indices_shape, &indices));
    Tensor* vv = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(2, vv_shape, &vv));
    
    // get the corresponding Eigen tensors for data access
    
    auto u_tensor = u.flat<double>().data();
    auto mu_tensor = mu.flat<double>().data();
    auto lamb_tensor = lamb.flat<double>().data();
    auto psi_tensor = psi->flat<double>().data();
    auto indices_tensor = indices->flat<int64>().data();
    auto vv_tensor = vv->flat<double>().data();   

    // implement your forward function here 

    // TODO:
    psi->flat<double>().setZero();
    NH_forward(psi_tensor, indices_tensor, vv_tensor, u_tensor, mu_tensor, lamb_tensor);
      
  }
};
REGISTER_KERNEL_BUILDER(Name("NeoHookean").Device(DEVICE_CPU), NeoHookeanOp);



class NeoHookeanGradOp : public OpKernel {
private:
  
public:
  explicit NeoHookeanGradOp(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_psi = context->input(0);
    const Tensor& grad_vv = context->input(1);
    const Tensor& psi = context->input(2);
    const Tensor& indices = context->input(3);
    const Tensor& vv = context->input(4);
    const Tensor& u = context->input(5);
    const Tensor& mu = context->input(6);
    const Tensor& lamb = context->input(7);
    
    
    const TensorShape& grad_psi_shape = grad_psi.shape();
    const TensorShape& grad_vv_shape = grad_vv.shape();
    const TensorShape& psi_shape = psi.shape();
    const TensorShape& indices_shape = indices.shape();
    const TensorShape& vv_shape = vv.shape();
    const TensorShape& u_shape = u.shape();
    const TensorShape& mu_shape = mu.shape();
    const TensorShape& lamb_shape = lamb.shape();
    
    
    DCHECK_EQ(grad_psi_shape.dims(), 1);
    DCHECK_EQ(grad_vv_shape.dims(), 1);
    DCHECK_EQ(psi_shape.dims(), 1);
    DCHECK_EQ(indices_shape.dims(), 2);
    DCHECK_EQ(vv_shape.dims(), 1);
    DCHECK_EQ(u_shape.dims(), 1);
    DCHECK_EQ(mu_shape.dims(), 1);
    DCHECK_EQ(lamb_shape.dims(), 1);

    // extra check
    // int m = Example.dim_size(0);
        
    // create output shape
    
    TensorShape grad_u_shape(u_shape);
    TensorShape grad_mu_shape(mu_shape);
    TensorShape grad_lamb_shape(lamb_shape);
            
    // create output tensor
    
    Tensor* grad_u = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_u_shape, &grad_u));
    Tensor* grad_mu = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, grad_mu_shape, &grad_mu));
    Tensor* grad_lamb = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(2, grad_lamb_shape, &grad_lamb));
    
    // get the corresponding Eigen tensors for data access
    
    auto u_tensor = u.flat<double>().data();
    auto mu_tensor = mu.flat<double>().data();
    auto lamb_tensor = lamb.flat<double>().data();
    auto grad_psi_tensor = grad_psi.flat<double>().data();
    auto grad_vv_tensor = grad_vv.flat<double>().data();
    auto psi_tensor = psi.flat<double>().data();
    auto indices_tensor = indices.flat<int64>().data();
    auto vv_tensor = vv.flat<double>().data();
    auto grad_u_tensor = grad_u->flat<double>().data();
    auto grad_mu_tensor = grad_mu->flat<double>().data();
    auto grad_lamb_tensor = grad_lamb->flat<double>().data();   

    // implement your backward function here 

    // TODO:
    
  }
};
REGISTER_KERNEL_BUILDER(Name("NeoHookeanGrad").Device(DEVICE_CPU), NeoHookeanGradOp);


/***************************************************************************************
**********************            GPU Operator            ******************************
***************************************************************************************/


#ifdef GOOGLE_CUDA
class NeoHookeanOpGPU : public OpKernel {
private:
  
public:
  explicit NeoHookeanOpGPU(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(3, context->num_inputs());
    
    
    const Tensor& u = context->input(0);
    const Tensor& mu = context->input(1);
    const Tensor& lamb = context->input(2);
    
    
    const TensorShape& u_shape = u.shape();
    const TensorShape& mu_shape = mu.shape();
    const TensorShape& lamb_shape = lamb.shape();
    
    
    DCHECK_EQ(u_shape.dims(), 1);
    DCHECK_EQ(mu_shape.dims(), 1);
    DCHECK_EQ(lamb_shape.dims(), 1);

    // extra check
        
    // create output shape
    
    TensorShape psi_shape({-1});
    TensorShape indices_shape({-1});
    TensorShape vv_shape({-1});
            
    // create output tensor
    
    Tensor* psi = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, psi_shape, &psi));
    Tensor* indices = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, indices_shape, &indices));
    Tensor* vv = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(2, vv_shape, &vv));
    
    // get the corresponding Eigen tensors for data access
    
    auto u_tensor = u.flat<double>().data();
    auto mu_tensor = mu.flat<double>().data();
    auto lamb_tensor = lamb.flat<double>().data();
    auto psi_tensor = psi->flat<double>().data();
    auto indices_tensor = indices->flat<int64>().data();
    auto vv_tensor = vv->flat<double>().data();   

    // implement your forward function here 

    // TODO:

  }
};
REGISTER_KERNEL_BUILDER(Name("NeoHookean").Device(DEVICE_GPU), NeoHookeanOpGPU);

class NeoHookeanGradOpGPU : public OpKernel {
private:
  
public:
  explicit NeoHookeanGradOpGPU(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_psi = context->input(0);
    const Tensor& grad_vv = context->input(1);
    const Tensor& psi = context->input(2);
    const Tensor& indices = context->input(3);
    const Tensor& vv = context->input(4);
    const Tensor& u = context->input(5);
    const Tensor& mu = context->input(6);
    const Tensor& lamb = context->input(7);
    
    
    const TensorShape& grad_psi_shape = grad_psi.shape();
    const TensorShape& grad_vv_shape = grad_vv.shape();
    const TensorShape& psi_shape = psi.shape();
    const TensorShape& indices_shape = indices.shape();
    const TensorShape& vv_shape = vv.shape();
    const TensorShape& u_shape = u.shape();
    const TensorShape& mu_shape = mu.shape();
    const TensorShape& lamb_shape = lamb.shape();
    
    
    DCHECK_EQ(grad_psi_shape.dims(), 1);
    DCHECK_EQ(grad_vv_shape.dims(), 1);
    DCHECK_EQ(psi_shape.dims(), 1);
    DCHECK_EQ(indices_shape.dims(), 1);
    DCHECK_EQ(vv_shape.dims(), 1);
    DCHECK_EQ(u_shape.dims(), 1);
    DCHECK_EQ(mu_shape.dims(), 1);
    DCHECK_EQ(lamb_shape.dims(), 1);

    // extra check
    // int m = Example.dim_size(0);
        
    // create output shape
    
    TensorShape grad_u_shape(u_shape);
    TensorShape grad_mu_shape(mu_shape);
    TensorShape grad_lamb_shape(lamb_shape);
            
    // create output tensor
    
    Tensor* grad_u = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_u_shape, &grad_u));
    Tensor* grad_mu = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, grad_mu_shape, &grad_mu));
    Tensor* grad_lamb = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(2, grad_lamb_shape, &grad_lamb));
    
    // get the corresponding Eigen tensors for data access
    
    auto u_tensor = u.flat<double>().data();
    auto mu_tensor = mu.flat<double>().data();
    auto lamb_tensor = lamb.flat<double>().data();
    auto grad_psi_tensor = grad_psi.flat<double>().data();
    auto grad_vv_tensor = grad_vv.flat<double>().data();
    auto psi_tensor = psi.flat<double>().data();
    auto indices_tensor = indices.flat<int64>().data();
    auto vv_tensor = vv.flat<double>().data();
    auto grad_u_tensor = grad_u->flat<double>().data();
    auto grad_mu_tensor = grad_mu->flat<double>().data();
    auto grad_lamb_tensor = grad_lamb->flat<double>().data();   

    // implement your backward function here 

    // TODO:
    
  }
};
REGISTER_KERNEL_BUILDER(Name("NeoHookeanGrad").Device(DEVICE_GPU), NeoHookeanGradOpGPU);

#endif