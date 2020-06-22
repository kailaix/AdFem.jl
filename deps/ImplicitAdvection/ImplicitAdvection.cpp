#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"
#include<cmath>

// Signatures for GPU kernels here 


using namespace tensorflow;
#include "ImplicitAdvection.h"


REGISTER_OP("ImplicitAdvection")
.Input("uv : double")
.Input("bc : int64")
.Input("bcval : double")
.Input("m : int64")
.Input("n : int64")
.Input("h : double")
.Output("ii : int64")
.Output("jj : int64")
.Output("vv : double")
.Output("rhs : double")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    
        shape_inference::ShapeHandle uv_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &uv_shape));
        shape_inference::ShapeHandle bc_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &bc_shape));
        shape_inference::ShapeHandle bcval_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &bcval_shape));
        shape_inference::ShapeHandle m_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &m_shape));
        shape_inference::ShapeHandle n_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 0, &n_shape));
        shape_inference::ShapeHandle h_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(5), 0, &h_shape));

        c->set_output(0, c->Vector(-1));
        c->set_output(1, c->Vector(-1));
        c->set_output(2, c->Vector(-1));
        c->set_output(3, c->Vector(-1));
    return Status::OK();
  });

REGISTER_OP("ImplicitAdvectionGrad")
.Input("grad_vv : double")
.Input("grad_rhs : double")
.Input("ii : int64")
.Input("jj : int64")
.Input("vv : double")
.Input("rhs : double")
.Input("uv : double")
.Input("bc : int64")
.Input("bcval : double")
.Input("m : int64")
.Input("n : int64")
.Input("h : double")
.Output("grad_uv : double")
.Output("grad_bc : int64")
.Output("grad_bcval : double")
.Output("grad_m : int64")
.Output("grad_n : int64")
.Output("grad_h : double");

/*-------------------------------------------------------------------------------------*/

class ImplicitAdvectionOp : public OpKernel {
private:
  
public:
  explicit ImplicitAdvectionOp(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(6, context->num_inputs());
    
    
    const Tensor& uv = context->input(0);
    const Tensor& bc = context->input(1);
    const Tensor& bcval = context->input(2);
    const Tensor& m = context->input(3);
    const Tensor& n = context->input(4);
    const Tensor& h = context->input(5);
    
    
    const TensorShape& uv_shape = uv.shape();
    const TensorShape& bc_shape = bc.shape();
    const TensorShape& bcval_shape = bcval.shape();
    const TensorShape& m_shape = m.shape();
    const TensorShape& n_shape = n.shape();
    const TensorShape& h_shape = h.shape();
    
    
    DCHECK_EQ(uv_shape.dims(), 1);
    DCHECK_EQ(bc_shape.dims(), 2);
    DCHECK_EQ(bcval_shape.dims(), 1);
    DCHECK_EQ(m_shape.dims(), 0);
    DCHECK_EQ(n_shape.dims(), 0);
    DCHECK_EQ(h_shape.dims(), 0);

    // extra check
        
    // create output shape
    auto uv_tensor = uv.flat<double>().data();
    auto bc_tensor = bc.flat<int64>().data();
    auto bcval_tensor = bcval.flat<double>().data();
    auto m_tensor = m.flat<int64>().data();
    auto n_tensor = n.flat<int64>().data();
    auto h_tensor = h.flat<double>().data();


    int m_ = *m_tensor, n_ = *n_tensor, nbc = bc_shape.dim_size(0);
    double h_ = *h_tensor;
    
    TensorShape rhs_shape({m_*n_});
    Tensor* rhs = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(3, rhs_shape, &rhs));
    auto rhs_tensor = rhs->flat<double>().data();   


    IAForward IAF;
    rhs->flat<double>().setZero();
    IAF.forward(rhs_tensor, bc_tensor, bcval_tensor, nbc, uv_tensor, uv_tensor+m_*n_, m_, n_, h_);
    
    int N = IAF.ii_.size();
    TensorShape ii_shape({N});
    TensorShape jj_shape({N});
    TensorShape vv_shape({N});
    
            
    // create output tensor
    
    Tensor* ii = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, ii_shape, &ii));
    Tensor* jj = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, jj_shape, &jj));
    Tensor* vv = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(2, vv_shape, &vv));
    
    
    // get the corresponding Eigen tensors for data access
    
    
    auto ii_tensor = ii->flat<int64>().data();
    auto jj_tensor = jj->flat<int64>().data();
    auto vv_tensor = vv->flat<double>().data();

    // implement your forward function here 

    // TODO:
    IAF.copy_data(ii_tensor, jj_tensor, vv_tensor);

  }
};
REGISTER_KERNEL_BUILDER(Name("ImplicitAdvection").Device(DEVICE_CPU), ImplicitAdvectionOp);



class ImplicitAdvectionGradOp : public OpKernel {
private:
  
public:
  explicit ImplicitAdvectionGradOp(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_vv = context->input(0);
    const Tensor& grad_rhs = context->input(1);
    const Tensor& ii = context->input(2);
    const Tensor& jj = context->input(3);
    const Tensor& vv = context->input(4);
    const Tensor& rhs = context->input(5);
    const Tensor& uv = context->input(6);
    const Tensor& bc = context->input(7);
    const Tensor& bcval = context->input(8);
    const Tensor& m = context->input(9);
    const Tensor& n = context->input(10);
    const Tensor& h = context->input(11);
    
    
    const TensorShape& grad_vv_shape = grad_vv.shape();
    const TensorShape& grad_rhs_shape = grad_rhs.shape();
    const TensorShape& ii_shape = ii.shape();
    const TensorShape& jj_shape = jj.shape();
    const TensorShape& vv_shape = vv.shape();
    const TensorShape& rhs_shape = rhs.shape();
    const TensorShape& uv_shape = uv.shape();
    const TensorShape& bc_shape = bc.shape();
    const TensorShape& bcval_shape = bcval.shape();
    const TensorShape& m_shape = m.shape();
    const TensorShape& n_shape = n.shape();
    const TensorShape& h_shape = h.shape();
    
    
    DCHECK_EQ(grad_vv_shape.dims(), 1);
    DCHECK_EQ(grad_rhs_shape.dims(), 1);
    DCHECK_EQ(ii_shape.dims(), 1);
    DCHECK_EQ(jj_shape.dims(), 1);
    DCHECK_EQ(vv_shape.dims(), 1);
    DCHECK_EQ(rhs_shape.dims(), 1);
    DCHECK_EQ(uv_shape.dims(), 1);
    DCHECK_EQ(bc_shape.dims(), 2);
    DCHECK_EQ(bcval_shape.dims(), 1);
    DCHECK_EQ(m_shape.dims(), 0);
    DCHECK_EQ(n_shape.dims(), 0);
    DCHECK_EQ(h_shape.dims(), 0);

    // extra check
    // int m = Example.dim_size(0);
        
    // create output shape
    
    TensorShape grad_uv_shape(uv_shape);
    TensorShape grad_bc_shape(bc_shape);
    TensorShape grad_bcval_shape(bcval_shape);
    TensorShape grad_m_shape(m_shape);
    TensorShape grad_n_shape(n_shape);
    TensorShape grad_h_shape(h_shape);
            
    // create output tensor
    
    Tensor* grad_uv = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_uv_shape, &grad_uv));
    Tensor* grad_bc = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, grad_bc_shape, &grad_bc));
    Tensor* grad_bcval = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(2, grad_bcval_shape, &grad_bcval));
    Tensor* grad_m = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(3, grad_m_shape, &grad_m));
    Tensor* grad_n = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(4, grad_n_shape, &grad_n));
    Tensor* grad_h = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(5, grad_h_shape, &grad_h));
    
    // get the corresponding Eigen tensors for data access
    
    auto uv_tensor = uv.flat<double>().data();
    auto bc_tensor = bc.flat<int64>().data();
    auto bcval_tensor = bcval.flat<double>().data();
    auto m_tensor = m.flat<int64>().data();
    auto n_tensor = n.flat<int64>().data();
    auto h_tensor = h.flat<double>().data();
    auto grad_vv_tensor = grad_vv.flat<double>().data();
    auto grad_rhs_tensor = grad_rhs.flat<double>().data();
    auto ii_tensor = ii.flat<int64>().data();
    auto jj_tensor = jj.flat<int64>().data();
    auto vv_tensor = vv.flat<double>().data();
    auto rhs_tensor = rhs.flat<double>().data();
    auto grad_uv_tensor = grad_uv->flat<double>().data();
    auto grad_bcval_tensor = grad_bcval->flat<double>().data();
    auto grad_h_tensor = grad_h->flat<double>().data();   

    // implement your backward function here 
    int m_ = *m_tensor, n_ = *n_tensor, nbc = bc_shape.dim_size(0);
    double h_ = *h_tensor;
    // TODO:
    grad_uv->flat<double>().setZero();
    grad_bcval->flat<double>().setZero();
    IA_backward(
      grad_bcval_tensor,  grad_uv_tensor, grad_uv_tensor + m_ * n_, 
      grad_vv_tensor, grad_rhs_tensor, bc_tensor, bcval_tensor, 
      nbc, uv_tensor, uv_tensor+m_*n_, m_, n_, h_
    );


  }
};
REGISTER_KERNEL_BUILDER(Name("ImplicitAdvectionGrad").Device(DEVICE_CPU), ImplicitAdvectionGradOp);


/***************************************************************************************
**********************            GPU Operator            ******************************
***************************************************************************************/


#ifdef GOOGLE_CUDA
class ImplicitAdvectionOpGPU : public OpKernel {
private:
  
public:
  explicit ImplicitAdvectionOpGPU(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(6, context->num_inputs());
    
    
    const Tensor& uv = context->input(0);
    const Tensor& bc = context->input(1);
    const Tensor& bcval = context->input(2);
    const Tensor& m = context->input(3);
    const Tensor& n = context->input(4);
    const Tensor& h = context->input(5);
    
    
    const TensorShape& uv_shape = uv.shape();
    const TensorShape& bc_shape = bc.shape();
    const TensorShape& bcval_shape = bcval.shape();
    const TensorShape& m_shape = m.shape();
    const TensorShape& n_shape = n.shape();
    const TensorShape& h_shape = h.shape();
    
    
    DCHECK_EQ(uv_shape.dims(), 1);
    DCHECK_EQ(bc_shape.dims(), 2);
    DCHECK_EQ(bcval_shape.dims(), 1);
    DCHECK_EQ(m_shape.dims(), 0);
    DCHECK_EQ(n_shape.dims(), 0);
    DCHECK_EQ(h_shape.dims(), 0);

    // extra check
        
    // create output shape
    
    TensorShape ii_shape({-1});
    TensorShape jj_shape({-1});
    TensorShape vv_shape({-1});
    TensorShape rhs_shape({-1});
            
    // create output tensor
    
    Tensor* ii = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, ii_shape, &ii));
    Tensor* jj = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, jj_shape, &jj));
    Tensor* vv = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(2, vv_shape, &vv));
    Tensor* rhs = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(3, rhs_shape, &rhs));
    
    // get the corresponding Eigen tensors for data access
    
    auto uv_tensor = uv.flat<double>().data();
    auto bc_tensor = bc.flat<int64>().data();
    auto bcval_tensor = bcval.flat<double>().data();
    auto m_tensor = m.flat<int64>().data();
    auto n_tensor = n.flat<int64>().data();
    auto h_tensor = h.flat<double>().data();
    auto ii_tensor = ii->flat<int64>().data();
    auto jj_tensor = jj->flat<int64>().data();
    auto vv_tensor = vv->flat<double>().data();
    auto rhs_tensor = rhs->flat<double>().data();   

    // implement your forward function here 

    // TODO:

  }
};
REGISTER_KERNEL_BUILDER(Name("ImplicitAdvection").Device(DEVICE_GPU), ImplicitAdvectionOpGPU);

class ImplicitAdvectionGradOpGPU : public OpKernel {
private:
  
public:
  explicit ImplicitAdvectionGradOpGPU(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_vv = context->input(0);
    const Tensor& grad_rhs = context->input(1);
    const Tensor& ii = context->input(2);
    const Tensor& jj = context->input(3);
    const Tensor& vv = context->input(4);
    const Tensor& rhs = context->input(5);
    const Tensor& uv = context->input(6);
    const Tensor& bc = context->input(7);
    const Tensor& bcval = context->input(8);
    const Tensor& m = context->input(9);
    const Tensor& n = context->input(10);
    const Tensor& h = context->input(11);
    
    
    const TensorShape& grad_vv_shape = grad_vv.shape();
    const TensorShape& grad_rhs_shape = grad_rhs.shape();
    const TensorShape& ii_shape = ii.shape();
    const TensorShape& jj_shape = jj.shape();
    const TensorShape& vv_shape = vv.shape();
    const TensorShape& rhs_shape = rhs.shape();
    const TensorShape& uv_shape = uv.shape();
    const TensorShape& bc_shape = bc.shape();
    const TensorShape& bcval_shape = bcval.shape();
    const TensorShape& m_shape = m.shape();
    const TensorShape& n_shape = n.shape();
    const TensorShape& h_shape = h.shape();
    
    
    DCHECK_EQ(grad_vv_shape.dims(), 1);
    DCHECK_EQ(grad_rhs_shape.dims(), 1);
    DCHECK_EQ(ii_shape.dims(), 1);
    DCHECK_EQ(jj_shape.dims(), 1);
    DCHECK_EQ(vv_shape.dims(), 1);
    DCHECK_EQ(rhs_shape.dims(), 1);
    DCHECK_EQ(uv_shape.dims(), 1);
    DCHECK_EQ(bc_shape.dims(), 2);
    DCHECK_EQ(bcval_shape.dims(), 1);
    DCHECK_EQ(m_shape.dims(), 0);
    DCHECK_EQ(n_shape.dims(), 0);
    DCHECK_EQ(h_shape.dims(), 0);

    // extra check
    // int m = Example.dim_size(0);
        
    // create output shape
    
    TensorShape grad_uv_shape(uv_shape);
    TensorShape grad_bc_shape(bc_shape);
    TensorShape grad_bcval_shape(bcval_shape);
    TensorShape grad_m_shape(m_shape);
    TensorShape grad_n_shape(n_shape);
    TensorShape grad_h_shape(h_shape);
            
    // create output tensor
    
    Tensor* grad_uv = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_uv_shape, &grad_uv));
    Tensor* grad_bc = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, grad_bc_shape, &grad_bc));
    Tensor* grad_bcval = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(2, grad_bcval_shape, &grad_bcval));
    Tensor* grad_m = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(3, grad_m_shape, &grad_m));
    Tensor* grad_n = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(4, grad_n_shape, &grad_n));
    Tensor* grad_h = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(5, grad_h_shape, &grad_h));
    
    // get the corresponding Eigen tensors for data access
    
    auto uv_tensor = uv.flat<double>().data();
    auto bc_tensor = bc.flat<int64>().data();
    auto bcval_tensor = bcval.flat<double>().data();
    auto m_tensor = m.flat<int64>().data();
    auto n_tensor = n.flat<int64>().data();
    auto h_tensor = h.flat<double>().data();
    auto grad_vv_tensor = grad_vv.flat<double>().data();
    auto grad_rhs_tensor = grad_rhs.flat<double>().data();
    auto ii_tensor = ii.flat<int64>().data();
    auto jj_tensor = jj.flat<int64>().data();
    auto vv_tensor = vv.flat<double>().data();
    auto rhs_tensor = rhs.flat<double>().data();
    auto grad_uv_tensor = grad_uv->flat<double>().data();
    auto grad_bcval_tensor = grad_bcval->flat<double>().data();
    auto grad_h_tensor = grad_h->flat<double>().data();   

    // implement your backward function here 

    // TODO:
    
  }
};
REGISTER_KERNEL_BUILDER(Name("ImplicitAdvectionGrad").Device(DEVICE_GPU), ImplicitAdvectionGradOpGPU);

#endif