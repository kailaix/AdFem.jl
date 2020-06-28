#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"
#include<cmath>

// Signatures for GPU kernels here 


using namespace tensorflow;
#include "FemAdvection.h"


REGISTER_OP("FemAdvection")
.Input("u : double")
.Input("v : double")
.Input("m : int64")
.Input("n : int64")
.Input("h : double")
.Output("ii : int64")
.Output("jj : int64")
.Output("vv : double")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    
        shape_inference::ShapeHandle u_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &u_shape));
        shape_inference::ShapeHandle v_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &v_shape));
        shape_inference::ShapeHandle m_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &m_shape));
        shape_inference::ShapeHandle n_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &n_shape));
        shape_inference::ShapeHandle h_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 0, &h_shape));

        c->set_output(0, c->Vector(-1));
        c->set_output(1, c->Vector(-1));
        c->set_output(2, c->Vector(-1));
    return Status::OK();
  });

REGISTER_OP("FemAdvectionGrad")
.Input("grad_vv : double")
.Input("ii : int64")
.Input("jj : int64")
.Input("vv : double")
.Input("u : double")
.Input("v : double")
.Input("m : int64")
.Input("n : int64")
.Input("h : double")
.Output("grad_u : double")
.Output("grad_v : double")
.Output("grad_m : int64")
.Output("grad_n : int64")
.Output("grad_h : double");

/*-------------------------------------------------------------------------------------*/

class FemAdvectionOp : public OpKernel {
private:
  
public:
  explicit FemAdvectionOp(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(5, context->num_inputs());
    
    
    const Tensor& u = context->input(0);
    const Tensor& v = context->input(1);
    const Tensor& m = context->input(2);
    const Tensor& n = context->input(3);
    const Tensor& h = context->input(4);
    
    
    const TensorShape& u_shape = u.shape();
    const TensorShape& v_shape = v.shape();
    const TensorShape& m_shape = m.shape();
    const TensorShape& n_shape = n.shape();
    const TensorShape& h_shape = h.shape();
    
    
    DCHECK_EQ(u_shape.dims(), 1);
    DCHECK_EQ(v_shape.dims(), 1);
    DCHECK_EQ(m_shape.dims(), 0);
    DCHECK_EQ(n_shape.dims(), 0);
    DCHECK_EQ(h_shape.dims(), 0);

    // extra check
        
    // create output shape
    auto m_tensor = m.flat<int64>().data();
    auto n_tensor = n.flat<int64>().data();

    int m_ = *m_tensor, n_ = *n_tensor; 
    TensorShape ii_shape({16*4*m_*n_});
    TensorShape jj_shape({16*4*m_*n_});
    TensorShape vv_shape({16*4*m_*n_});
            
    // create output tensor
    
    Tensor* ii = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, ii_shape, &ii));
    Tensor* jj = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, jj_shape, &jj));
    Tensor* vv = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(2, vv_shape, &vv));
    
    // get the corresponding Eigen tensors for data access
    
    auto u_tensor = u.flat<double>().data();
    auto v_tensor = v.flat<double>().data();
    
    auto h_tensor = h.flat<double>().data();
    auto ii_tensor = ii->flat<int64>().data();
    auto jj_tensor = jj->flat<int64>().data();
    auto vv_tensor = vv->flat<double>().data();   

    // implement your forward function here 

    // TODO:
    FemAdvection_forward(ii_tensor, jj_tensor, vv_tensor, u_tensor, v_tensor, m_, n_, *h_tensor);
  }
};
REGISTER_KERNEL_BUILDER(Name("FemAdvection").Device(DEVICE_CPU), FemAdvectionOp);



class FemAdvectionGradOp : public OpKernel {
private:
  
public:
  explicit FemAdvectionGradOp(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_vv = context->input(0);
    const Tensor& ii = context->input(1);
    const Tensor& jj = context->input(2);
    const Tensor& vv = context->input(3);
    const Tensor& u = context->input(4);
    const Tensor& v = context->input(5);
    const Tensor& m = context->input(6);
    const Tensor& n = context->input(7);
    const Tensor& h = context->input(8);
    
    
    const TensorShape& grad_vv_shape = grad_vv.shape();
    const TensorShape& ii_shape = ii.shape();
    const TensorShape& jj_shape = jj.shape();
    const TensorShape& vv_shape = vv.shape();
    const TensorShape& u_shape = u.shape();
    const TensorShape& v_shape = v.shape();
    const TensorShape& m_shape = m.shape();
    const TensorShape& n_shape = n.shape();
    const TensorShape& h_shape = h.shape();
    
    
    DCHECK_EQ(grad_vv_shape.dims(), 1);
    DCHECK_EQ(ii_shape.dims(), 1);
    DCHECK_EQ(jj_shape.dims(), 1);
    DCHECK_EQ(vv_shape.dims(), 1);
    DCHECK_EQ(u_shape.dims(), 1);
    DCHECK_EQ(v_shape.dims(), 1);
    DCHECK_EQ(m_shape.dims(), 0);
    DCHECK_EQ(n_shape.dims(), 0);
    DCHECK_EQ(h_shape.dims(), 0);

    // extra check
    // int m = Example.dim_size(0);
        
    // create output shape
    
    TensorShape grad_u_shape(u_shape);
    TensorShape grad_v_shape(v_shape);
    TensorShape grad_m_shape(m_shape);
    TensorShape grad_n_shape(n_shape);
    TensorShape grad_h_shape(h_shape);
            
    // create output tensor
    
    Tensor* grad_u = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_u_shape, &grad_u));
    Tensor* grad_v = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, grad_v_shape, &grad_v));
    Tensor* grad_m = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(2, grad_m_shape, &grad_m));
    Tensor* grad_n = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(3, grad_n_shape, &grad_n));
    Tensor* grad_h = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(4, grad_h_shape, &grad_h));
    
    // get the corresponding Eigen tensors for data access
    
    auto u_tensor = u.flat<double>().data();
    auto v_tensor = v.flat<double>().data();
    auto m_tensor = m.flat<int64>().data();
    auto n_tensor = n.flat<int64>().data();
    auto h_tensor = h.flat<double>().data();
    auto grad_vv_tensor = grad_vv.flat<double>().data();
    auto ii_tensor = ii.flat<int64>().data();
    auto jj_tensor = jj.flat<int64>().data();
    auto vv_tensor = vv.flat<double>().data();
    auto grad_u_tensor = grad_u->flat<double>().data();
    auto grad_v_tensor = grad_v->flat<double>().data();
    auto grad_h_tensor = grad_h->flat<double>().data();   

    // implement your backward function here 

    // TODO:
    grad_u->flat<double>().setZero();
    grad_v->flat<double>().setZero();
    FemAdvection_backward(grad_u_tensor, grad_v_tensor, grad_vv_tensor, *m_tensor, *n_tensor, *h_tensor);
    
  }
};
REGISTER_KERNEL_BUILDER(Name("FemAdvectionGrad").Device(DEVICE_CPU), FemAdvectionGradOp);


/***************************************************************************************
**********************            GPU Operator            ******************************
***************************************************************************************/


#ifdef GOOGLE_CUDA
class FemAdvectionOpGPU : public OpKernel {
private:
  
public:
  explicit FemAdvectionOpGPU(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(5, context->num_inputs());
    
    
    const Tensor& u = context->input(0);
    const Tensor& v = context->input(1);
    const Tensor& m = context->input(2);
    const Tensor& n = context->input(3);
    const Tensor& h = context->input(4);
    
    
    const TensorShape& u_shape = u.shape();
    const TensorShape& v_shape = v.shape();
    const TensorShape& m_shape = m.shape();
    const TensorShape& n_shape = n.shape();
    const TensorShape& h_shape = h.shape();
    
    
    DCHECK_EQ(u_shape.dims(), 1);
    DCHECK_EQ(v_shape.dims(), 1);
    DCHECK_EQ(m_shape.dims(), 0);
    DCHECK_EQ(n_shape.dims(), 0);
    DCHECK_EQ(h_shape.dims(), 0);

    // extra check
        
    // create output shape
    
    TensorShape ii_shape({-1});
    TensorShape jj_shape({-1});
    TensorShape vv_shape({-1});
            
    // create output tensor
    
    Tensor* ii = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, ii_shape, &ii));
    Tensor* jj = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, jj_shape, &jj));
    Tensor* vv = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(2, vv_shape, &vv));
    
    // get the corresponding Eigen tensors for data access
    
    auto u_tensor = u.flat<double>().data();
    auto v_tensor = v.flat<double>().data();
    auto m_tensor = m.flat<int64>().data();
    auto n_tensor = n.flat<int64>().data();
    auto h_tensor = h.flat<double>().data();
    auto ii_tensor = ii->flat<int64>().data();
    auto jj_tensor = jj->flat<int64>().data();
    auto vv_tensor = vv->flat<double>().data();   

    // implement your forward function here 

    // TODO:

  }
};
REGISTER_KERNEL_BUILDER(Name("FemAdvection").Device(DEVICE_GPU), FemAdvectionOpGPU);

class FemAdvectionGradOpGPU : public OpKernel {
private:
  
public:
  explicit FemAdvectionGradOpGPU(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_vv = context->input(0);
    const Tensor& ii = context->input(1);
    const Tensor& jj = context->input(2);
    const Tensor& vv = context->input(3);
    const Tensor& u = context->input(4);
    const Tensor& v = context->input(5);
    const Tensor& m = context->input(6);
    const Tensor& n = context->input(7);
    const Tensor& h = context->input(8);
    
    
    const TensorShape& grad_vv_shape = grad_vv.shape();
    const TensorShape& ii_shape = ii.shape();
    const TensorShape& jj_shape = jj.shape();
    const TensorShape& vv_shape = vv.shape();
    const TensorShape& u_shape = u.shape();
    const TensorShape& v_shape = v.shape();
    const TensorShape& m_shape = m.shape();
    const TensorShape& n_shape = n.shape();
    const TensorShape& h_shape = h.shape();
    
    
    DCHECK_EQ(grad_vv_shape.dims(), 1);
    DCHECK_EQ(ii_shape.dims(), 1);
    DCHECK_EQ(jj_shape.dims(), 1);
    DCHECK_EQ(vv_shape.dims(), 1);
    DCHECK_EQ(u_shape.dims(), 1);
    DCHECK_EQ(v_shape.dims(), 1);
    DCHECK_EQ(m_shape.dims(), 0);
    DCHECK_EQ(n_shape.dims(), 0);
    DCHECK_EQ(h_shape.dims(), 0);

    // extra check
    // int m = Example.dim_size(0);
        
    // create output shape
    
    TensorShape grad_u_shape(u_shape);
    TensorShape grad_v_shape(v_shape);
    TensorShape grad_m_shape(m_shape);
    TensorShape grad_n_shape(n_shape);
    TensorShape grad_h_shape(h_shape);
            
    // create output tensor
    
    Tensor* grad_u = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_u_shape, &grad_u));
    Tensor* grad_v = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, grad_v_shape, &grad_v));
    Tensor* grad_m = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(2, grad_m_shape, &grad_m));
    Tensor* grad_n = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(3, grad_n_shape, &grad_n));
    Tensor* grad_h = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(4, grad_h_shape, &grad_h));
    
    // get the corresponding Eigen tensors for data access
    
    auto u_tensor = u.flat<double>().data();
    auto v_tensor = v.flat<double>().data();
    auto m_tensor = m.flat<int64>().data();
    auto n_tensor = n.flat<int64>().data();
    auto h_tensor = h.flat<double>().data();
    auto grad_vv_tensor = grad_vv.flat<double>().data();
    auto ii_tensor = ii.flat<int64>().data();
    auto jj_tensor = jj.flat<int64>().data();
    auto vv_tensor = vv.flat<double>().data();
    auto grad_u_tensor = grad_u->flat<double>().data();
    auto grad_v_tensor = grad_v->flat<double>().data();
    auto grad_h_tensor = grad_h->flat<double>().data();   

    // implement your backward function here 

    // TODO:
    
  }
};
REGISTER_KERNEL_BUILDER(Name("FemAdvectionGrad").Device(DEVICE_GPU), FemAdvectionGradOpGPU);

#endif