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
.Output("ic : double")
.Output("jc : double")
.Output("indices1 : int64")
.Output("vv1 : double")
.Output("indices2 : int64")
.Output("vv2 : double")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    
        shape_inference::ShapeHandle u_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &u_shape));
        shape_inference::ShapeHandle mu_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &mu_shape));
        shape_inference::ShapeHandle lamb_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &lamb_shape));

        c->set_output(0, c->Vector(-1));
        c->set_output(1, c->Vector(-1));
        c->set_output(2, c->Vector(-1));
        c->set_output(3, c->Vector(-1));
        c->set_output(4, c->Vector(-1));
        c->set_output(5, c->Vector(-1));
    return Status::OK();
  });

REGISTER_OP("NeoHookeanGrad")
.Input("grad_ic : double")
.Input("grad_jc : double")
.Input("grad_vv1 : double")
.Input("grad_vv2 : double")
.Input("ic : double")
.Input("jc : double")
.Input("indices1 : int64")
.Input("vv1 : double")
.Input("indices2 : int64")
.Input("vv2 : double")
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
    
    TensorShape ic_shape({2*mmesh.ndof});
    TensorShape jc_shape({2*mmesh.ndof});
    TensorShape indices1_shape({mmesh.ngauss * (2*mmesh.elem_ndof) * (2*mmesh.elem_ndof), 2});
    TensorShape vv1_shape({mmesh.ngauss * (2*mmesh.elem_ndof) * (2*mmesh.elem_ndof)});
    TensorShape indices2_shape({mmesh.ngauss * (2*mmesh.elem_ndof) * (2*mmesh.elem_ndof), 2});
    TensorShape vv2_shape({mmesh.ngauss * (2*mmesh.elem_ndof) * (2*mmesh.elem_ndof)});
            
    // create output tensor
    
    Tensor* ic = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, ic_shape, &ic));
    Tensor* jc = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, jc_shape, &jc));
    Tensor* indices1 = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(2, indices1_shape, &indices1));
    Tensor* vv1 = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(3, vv1_shape, &vv1));
    Tensor* indices2 = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(4, indices2_shape, &indices2));
    Tensor* vv2 = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(5, vv2_shape, &vv2));
    
    // get the corresponding Eigen tensors for data access
    
    auto u_tensor = u.flat<double>().data();
    auto mu_tensor = mu.flat<double>().data();
    auto lamb_tensor = lamb.flat<double>().data();
    auto ic_tensor = ic->flat<double>().data();
    auto jc_tensor = jc->flat<double>().data();
    auto indices1_tensor = indices1->flat<int64>().data();
    auto vv1_tensor = vv1->flat<double>().data();
    auto indices2_tensor = indices2->flat<int64>().data();
    auto vv2_tensor = vv2->flat<double>().data();   

    // implement your forward function here 

    // TODO:
    NH_forward(ic_tensor, jc_tensor, indices1_tensor, vv1_tensor, indices2_tensor, vv2_tensor, u_tensor,
            mu_tensor, lamb_tensor);


  }
};
REGISTER_KERNEL_BUILDER(Name("NeoHookean").Device(DEVICE_CPU), NeoHookeanOp);



class NeoHookeanGradOp : public OpKernel {
private:
  
public:
  explicit NeoHookeanGradOp(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_ic = context->input(0);
    const Tensor& grad_jc = context->input(1);
    const Tensor& grad_vv1 = context->input(2);
    const Tensor& grad_vv2 = context->input(3);
    const Tensor& ic = context->input(4);
    const Tensor& jc = context->input(5);
    const Tensor& indices1 = context->input(6);
    const Tensor& vv1 = context->input(7);
    const Tensor& indices2 = context->input(8);
    const Tensor& vv2 = context->input(9);
    const Tensor& u = context->input(10);
    const Tensor& mu = context->input(11);
    const Tensor& lamb = context->input(12);
    
    
    const TensorShape& grad_ic_shape = grad_ic.shape();
    const TensorShape& grad_jc_shape = grad_jc.shape();
    const TensorShape& grad_vv1_shape = grad_vv1.shape();
    const TensorShape& grad_vv2_shape = grad_vv2.shape();
    const TensorShape& ic_shape = ic.shape();
    const TensorShape& jc_shape = jc.shape();
    const TensorShape& indices1_shape = indices1.shape();
    const TensorShape& vv1_shape = vv1.shape();
    const TensorShape& indices2_shape = indices2.shape();
    const TensorShape& vv2_shape = vv2.shape();
    const TensorShape& u_shape = u.shape();
    const TensorShape& mu_shape = mu.shape();
    const TensorShape& lamb_shape = lamb.shape();
    
    
    DCHECK_EQ(grad_ic_shape.dims(), 1);
    DCHECK_EQ(grad_jc_shape.dims(), 1);
    DCHECK_EQ(grad_vv1_shape.dims(), 1);
    DCHECK_EQ(grad_vv2_shape.dims(), 1);
    DCHECK_EQ(ic_shape.dims(), 1);
    DCHECK_EQ(jc_shape.dims(), 1);
    DCHECK_EQ(indices1_shape.dims(), 1);
    DCHECK_EQ(vv1_shape.dims(), 1);
    DCHECK_EQ(indices2_shape.dims(), 1);
    DCHECK_EQ(vv2_shape.dims(), 1);
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
    auto grad_ic_tensor = grad_ic.flat<double>().data();
    auto grad_jc_tensor = grad_jc.flat<double>().data();
    auto grad_vv1_tensor = grad_vv1.flat<double>().data();
    auto grad_vv2_tensor = grad_vv2.flat<double>().data();
    auto ic_tensor = ic.flat<double>().data();
    auto jc_tensor = jc.flat<double>().data();
    auto indices1_tensor = indices1.flat<int64>().data();
    auto vv1_tensor = vv1.flat<double>().data();
    auto indices2_tensor = indices2.flat<int64>().data();
    auto vv2_tensor = vv2.flat<double>().data();
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
    
    TensorShape ic_shape({-1});
    TensorShape jc_shape({-1});
    TensorShape indices1_shape({-1});
    TensorShape vv1_shape({-1});
    TensorShape indices2_shape({-1});
    TensorShape vv2_shape({-1});
            
    // create output tensor
    
    Tensor* ic = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, ic_shape, &ic));
    Tensor* jc = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, jc_shape, &jc));
    Tensor* indices1 = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(2, indices1_shape, &indices1));
    Tensor* vv1 = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(3, vv1_shape, &vv1));
    Tensor* indices2 = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(4, indices2_shape, &indices2));
    Tensor* vv2 = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(5, vv2_shape, &vv2));
    
    // get the corresponding Eigen tensors for data access
    
    auto u_tensor = u.flat<double>().data();
    auto mu_tensor = mu.flat<double>().data();
    auto lamb_tensor = lamb.flat<double>().data();
    auto ic_tensor = ic->flat<double>().data();
    auto jc_tensor = jc->flat<double>().data();
    auto indices1_tensor = indices1->flat<int64>().data();
    auto vv1_tensor = vv1->flat<double>().data();
    auto indices2_tensor = indices2->flat<int64>().data();
    auto vv2_tensor = vv2->flat<double>().data();   

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
    
    
    const Tensor& grad_ic = context->input(0);
    const Tensor& grad_jc = context->input(1);
    const Tensor& grad_vv1 = context->input(2);
    const Tensor& grad_vv2 = context->input(3);
    const Tensor& ic = context->input(4);
    const Tensor& jc = context->input(5);
    const Tensor& indices1 = context->input(6);
    const Tensor& vv1 = context->input(7);
    const Tensor& indices2 = context->input(8);
    const Tensor& vv2 = context->input(9);
    const Tensor& u = context->input(10);
    const Tensor& mu = context->input(11);
    const Tensor& lamb = context->input(12);
    
    
    const TensorShape& grad_ic_shape = grad_ic.shape();
    const TensorShape& grad_jc_shape = grad_jc.shape();
    const TensorShape& grad_vv1_shape = grad_vv1.shape();
    const TensorShape& grad_vv2_shape = grad_vv2.shape();
    const TensorShape& ic_shape = ic.shape();
    const TensorShape& jc_shape = jc.shape();
    const TensorShape& indices1_shape = indices1.shape();
    const TensorShape& vv1_shape = vv1.shape();
    const TensorShape& indices2_shape = indices2.shape();
    const TensorShape& vv2_shape = vv2.shape();
    const TensorShape& u_shape = u.shape();
    const TensorShape& mu_shape = mu.shape();
    const TensorShape& lamb_shape = lamb.shape();
    
    
    DCHECK_EQ(grad_ic_shape.dims(), 1);
    DCHECK_EQ(grad_jc_shape.dims(), 1);
    DCHECK_EQ(grad_vv1_shape.dims(), 1);
    DCHECK_EQ(grad_vv2_shape.dims(), 1);
    DCHECK_EQ(ic_shape.dims(), 1);
    DCHECK_EQ(jc_shape.dims(), 1);
    DCHECK_EQ(indices1_shape.dims(), 1);
    DCHECK_EQ(vv1_shape.dims(), 1);
    DCHECK_EQ(indices2_shape.dims(), 1);
    DCHECK_EQ(vv2_shape.dims(), 1);
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
    auto grad_ic_tensor = grad_ic.flat<double>().data();
    auto grad_jc_tensor = grad_jc.flat<double>().data();
    auto grad_vv1_tensor = grad_vv1.flat<double>().data();
    auto grad_vv2_tensor = grad_vv2.flat<double>().data();
    auto ic_tensor = ic.flat<double>().data();
    auto jc_tensor = jc.flat<double>().data();
    auto indices1_tensor = indices1.flat<int64>().data();
    auto vv1_tensor = vv1.flat<double>().data();
    auto indices2_tensor = indices2.flat<int64>().data();
    auto vv2_tensor = vv2.flat<double>().data();
    auto grad_u_tensor = grad_u->flat<double>().data();
    auto grad_mu_tensor = grad_mu->flat<double>().data();
    auto grad_lamb_tensor = grad_lamb->flat<double>().data();   

    // implement your backward function here 

    // TODO:
    
  }
};
REGISTER_KERNEL_BUILDER(Name("NeoHookeanGrad").Device(DEVICE_GPU), NeoHookeanGradOpGPU);

#endif