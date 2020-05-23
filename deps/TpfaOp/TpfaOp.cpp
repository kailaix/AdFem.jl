#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"
#include<cmath>

// Signatures for GPU kernels here 


using namespace tensorflow;
#include "TpfaOp.h"


REGISTER_OP("TpfaOp")
.Input("kvalue : double")
.Input("bc : int64")
.Input("pval : double")
.Input("m : int64")
.Input("n : int64")
.Input("h : double")
.Output("ii : int64")
.Output("jj : int64")
.Output("vv : double")
.Output("rhs : double")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    
        shape_inference::ShapeHandle kvalue_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &kvalue_shape));
        shape_inference::ShapeHandle bc_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &bc_shape));
        shape_inference::ShapeHandle pval_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &pval_shape));
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

REGISTER_OP("TpfaOpGrad")
.Input("grad_vv : double")
.Input("grad_rhs : double")
.Input("ii : int64")
.Input("jj : int64")
.Input("vv : double")
.Input("rhs : double")
.Input("kvalue : double")
.Input("bc : int64")
.Input("pval : double")
.Input("m : int64")
.Input("n : int64")
.Input("h : double")
.Output("grad_kvalue : double")
.Output("grad_bc : int64")
.Output("grad_pval : double")
.Output("grad_m : int64")
.Output("grad_n : int64")
.Output("grad_h : double");

/*-------------------------------------------------------------------------------------*/

class TpfaOpOp : public OpKernel {
private:
  
public:
  explicit TpfaOpOp(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(6, context->num_inputs());
    
    
    const Tensor& kvalue = context->input(0);
    const Tensor& bc = context->input(1);
    const Tensor& pval = context->input(2);
    const Tensor& m = context->input(3);
    const Tensor& n = context->input(4);
    const Tensor& h = context->input(5);
    
    
    const TensorShape& kvalue_shape = kvalue.shape();
    const TensorShape& bc_shape = bc.shape();
    const TensorShape& pval_shape = pval.shape();
    const TensorShape& m_shape = m.shape();
    const TensorShape& n_shape = n.shape();
    const TensorShape& h_shape = h.shape();
    
    
    DCHECK_EQ(kvalue_shape.dims(), 1);
    DCHECK_EQ(bc_shape.dims(), 2);
    DCHECK_EQ(pval_shape.dims(), 1);
    DCHECK_EQ(m_shape.dims(), 0);
    DCHECK_EQ(n_shape.dims(), 0);
    DCHECK_EQ(h_shape.dims(), 0);

    // extra check
        
    // create output shape
    int N_ = kvalue_shape.dim_size(0);
    int N = bc_shape.dim_size(0);
    TensorShape rhs_shape({N_});
            
    // create output tensor
    
    Tensor* rhs = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(3, rhs_shape, &rhs));
    
    // get the corresponding Eigen tensors for data access
    
    auto kvalue_tensor = kvalue.flat<double>().data();
    auto bc_tensor = bc.flat<int64>().data();
    auto pval_tensor = pval.flat<double>().data();
    auto m_tensor = m.flat<int64>().data();
    auto n_tensor = n.flat<int64>().data();
    auto h_tensor = h.flat<double>().data();
    auto rhs_tensor = rhs->flat<double>().data();   

    // implement your forward function here 

    // TODO:
    rhs->flat<double>().setZero();
    forward(context, rhs_tensor, kvalue_tensor, bc_tensor, pval_tensor, 
        N, *m_tensor, *n_tensor, *h_tensor);

  }
};
REGISTER_KERNEL_BUILDER(Name("TpfaOp").Device(DEVICE_CPU), TpfaOpOp);



class TpfaOpGradOp : public OpKernel {
private:
  
public:
  explicit TpfaOpGradOp(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_vv = context->input(0);
    const Tensor& grad_rhs = context->input(1);
    const Tensor& ii = context->input(2);
    const Tensor& jj = context->input(3);
    const Tensor& vv = context->input(4);
    const Tensor& rhs = context->input(5);
    const Tensor& kvalue = context->input(6);
    const Tensor& bc = context->input(7);
    const Tensor& pval = context->input(8);
    const Tensor& m = context->input(9);
    const Tensor& n = context->input(10);
    const Tensor& h = context->input(11);
    
    
    const TensorShape& grad_vv_shape = grad_vv.shape();
    const TensorShape& grad_rhs_shape = grad_rhs.shape();
    const TensorShape& ii_shape = ii.shape();
    const TensorShape& jj_shape = jj.shape();
    const TensorShape& vv_shape = vv.shape();
    const TensorShape& rhs_shape = rhs.shape();
    const TensorShape& kvalue_shape = kvalue.shape();
    const TensorShape& bc_shape = bc.shape();
    const TensorShape& pval_shape = pval.shape();
    const TensorShape& m_shape = m.shape();
    const TensorShape& n_shape = n.shape();
    const TensorShape& h_shape = h.shape();
    
    
    DCHECK_EQ(grad_vv_shape.dims(), 1);
    DCHECK_EQ(grad_rhs_shape.dims(), 1);
    DCHECK_EQ(ii_shape.dims(), 1);
    DCHECK_EQ(jj_shape.dims(), 1);
    DCHECK_EQ(vv_shape.dims(), 1);
    DCHECK_EQ(rhs_shape.dims(), 1);
    DCHECK_EQ(kvalue_shape.dims(), 1);
    DCHECK_EQ(bc_shape.dims(), 2);
    DCHECK_EQ(pval_shape.dims(), 1);
    DCHECK_EQ(m_shape.dims(), 0);
    DCHECK_EQ(n_shape.dims(), 0);
    DCHECK_EQ(h_shape.dims(), 0);

    // extra check
    // int m = Example.dim_size(0);
        
    // create output shape
    
    TensorShape grad_kvalue_shape(kvalue_shape);
    TensorShape grad_bc_shape(bc_shape);
    TensorShape grad_pval_shape(pval_shape);
    TensorShape grad_m_shape(m_shape);
    TensorShape grad_n_shape(n_shape);
    TensorShape grad_h_shape(h_shape);
            
    // create output tensor
    
    Tensor* grad_kvalue = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_kvalue_shape, &grad_kvalue));
    Tensor* grad_bc = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, grad_bc_shape, &grad_bc));
    Tensor* grad_pval = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(2, grad_pval_shape, &grad_pval));
    Tensor* grad_m = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(3, grad_m_shape, &grad_m));
    Tensor* grad_n = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(4, grad_n_shape, &grad_n));
    Tensor* grad_h = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(5, grad_h_shape, &grad_h));
    
    // get the corresponding Eigen tensors for data access
    
    auto kvalue_tensor = kvalue.flat<double>().data();
    auto bc_tensor = bc.flat<int64>().data();
    auto pval_tensor = pval.flat<double>().data();
    auto m_tensor = m.flat<int64>().data();
    auto n_tensor = n.flat<int64>().data();
    auto h_tensor = h.flat<double>().data();
    auto grad_vv_tensor = grad_vv.flat<double>().data();
    auto grad_rhs_tensor = grad_rhs.flat<double>().data();
    auto ii_tensor = ii.flat<int64>().data();
    auto jj_tensor = jj.flat<int64>().data();
    auto vv_tensor = vv.flat<double>().data();
    auto rhs_tensor = rhs.flat<double>().data();
    auto grad_kvalue_tensor = grad_kvalue->flat<double>().data();
    auto grad_pval_tensor = grad_pval->flat<double>().data();
    auto grad_h_tensor = grad_h->flat<double>().data();   

    // implement your backward function here 

    // TODO:
    int N = bc_shape.dim_size(0);
    grad_pval->flat<double>().setZero();
    grad_kvalue->flat<double>().setZero();
    backward(grad_pval_tensor, grad_kvalue_tensor, grad_vv_tensor, grad_rhs_tensor, 
        rhs_tensor, kvalue_tensor, bc_tensor, pval_tensor, N, *m_tensor, *n_tensor, *h_tensor);
  }
};
REGISTER_KERNEL_BUILDER(Name("TpfaOpGrad").Device(DEVICE_CPU), TpfaOpGradOp);
