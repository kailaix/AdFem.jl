#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"
#include<cmath>

// Signatures for GPU kernels here 


using namespace tensorflow;
#include "InteractionTerm.h"


REGISTER_OP("InteractionTerm")
.Input("p : double")
.Input("m : int64")
.Input("n : int64")
.Input("h : double")
.Output("out : double")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    
        shape_inference::ShapeHandle p_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &p_shape));
        shape_inference::ShapeHandle m_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &m_shape));
        shape_inference::ShapeHandle n_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &n_shape));
        shape_inference::ShapeHandle h_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &h_shape));

        c->set_output(0, c->Vector(-1));
    return Status::OK();
  });

REGISTER_OP("InteractionTermGrad")
.Input("grad_out : double")
.Input("out : double")
.Input("p : double")
.Input("m : int64")
.Input("n : int64")
.Input("h : double")
.Output("grad_p : double")
.Output("grad_m : int64")
.Output("grad_n : int64")
.Output("grad_h : double");

/*-------------------------------------------------------------------------------------*/

class InteractionTermOp : public OpKernel {
private:
  
public:
  explicit InteractionTermOp(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(4, context->num_inputs());
    
    
    const Tensor& p = context->input(0);
    const Tensor& m = context->input(1);
    const Tensor& n = context->input(2);
    const Tensor& h = context->input(3);
    
    
    const TensorShape& p_shape = p.shape();
    const TensorShape& m_shape = m.shape();
    const TensorShape& n_shape = n.shape();
    const TensorShape& h_shape = h.shape();
    
    
    DCHECK_EQ(p_shape.dims(), 1);
    DCHECK_EQ(m_shape.dims(), 0);
    DCHECK_EQ(n_shape.dims(), 0);
    DCHECK_EQ(h_shape.dims(), 0);

    // extra check
        
    // create output shape
    auto m_tensor = m.flat<int64>().data();
    auto n_tensor = n.flat<int64>().data();
    int m_ = *m_tensor, n_ = *n_tensor;
    TensorShape out_shape({2*(m_+1)*(n_+1)});
            
    // create output tensor
    
    Tensor* out = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &out));
    
    // get the corresponding Eigen tensors for data access
    
    auto p_tensor = p.flat<double>().data();
    
    auto h_tensor = h.flat<double>().data();
    auto out_tensor = out->flat<double>().data();   

    // implement your forward function here 

    // TODO:
    out->flat<double>().setZero();
    InteractionTerm_forward(out_tensor, p_tensor, m_, n_, *h_tensor);

  }
};
REGISTER_KERNEL_BUILDER(Name("InteractionTerm").Device(DEVICE_CPU), InteractionTermOp);



class InteractionTermGradOp : public OpKernel {
private:
  
public:
  explicit InteractionTermGradOp(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_out = context->input(0);
    const Tensor& out = context->input(1);
    const Tensor& p = context->input(2);
    const Tensor& m = context->input(3);
    const Tensor& n = context->input(4);
    const Tensor& h = context->input(5);
    
    
    const TensorShape& grad_out_shape = grad_out.shape();
    const TensorShape& out_shape = out.shape();
    const TensorShape& p_shape = p.shape();
    const TensorShape& m_shape = m.shape();
    const TensorShape& n_shape = n.shape();
    const TensorShape& h_shape = h.shape();
    
    
    DCHECK_EQ(grad_out_shape.dims(), 1);
    DCHECK_EQ(out_shape.dims(), 1);
    DCHECK_EQ(p_shape.dims(), 1);
    DCHECK_EQ(m_shape.dims(), 0);
    DCHECK_EQ(n_shape.dims(), 0);
    DCHECK_EQ(h_shape.dims(), 0);

    // extra check
    // int m = Example.dim_size(0);
        
    // create output shape
    
    TensorShape grad_p_shape(p_shape);
    TensorShape grad_m_shape(m_shape);
    TensorShape grad_n_shape(n_shape);
    TensorShape grad_h_shape(h_shape);
            
    // create output tensor
    
    Tensor* grad_p = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_p_shape, &grad_p));
    Tensor* grad_m = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, grad_m_shape, &grad_m));
    Tensor* grad_n = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(2, grad_n_shape, &grad_n));
    Tensor* grad_h = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(3, grad_h_shape, &grad_h));
    
    // get the corresponding Eigen tensors for data access
    
    auto p_tensor = p.flat<double>().data();
    auto m_tensor = m.flat<int64>().data();
    auto n_tensor = n.flat<int64>().data();
    auto h_tensor = h.flat<double>().data();
    auto grad_out_tensor = grad_out.flat<double>().data();
    auto out_tensor = out.flat<double>().data();
    auto grad_p_tensor = grad_p->flat<double>().data();
    auto grad_h_tensor = grad_h->flat<double>().data();   

    // implement your backward function here 

    // TODO:
    grad_p->flat<double>().setZero();
    InteractionTerm_backward(grad_p_tensor, grad_out_tensor, 
      out_tensor, p_tensor, *m_tensor, *n_tensor, *h_tensor);
    
  }
};
REGISTER_KERNEL_BUILDER(Name("InteractionTermGrad").Device(DEVICE_CPU), InteractionTermGradOp);
