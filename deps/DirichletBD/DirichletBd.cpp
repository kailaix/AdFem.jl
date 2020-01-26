#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"
#include<cmath>

#ifdef USE_GPU
#include "tensorflow/core/util/gpu_kernel_helper.h"
namespace tensorflow{
  typedef Eigen::GpuDevice GPUDevice;
  void forwardGPU(const GPUDevice &d);
  void backwardGPU(const GPUDevice &d);
}
#endif

using namespace tensorflow;
#include "DirichletBd.h"

REGISTER_OP("DirichletBd")

.Input("ii : int64")
.Input("jj : int64")
.Input("vv : double")
.Input("bd : int32")
.Input("m : int32")
.Input("n : int32")
.Input("h : double")
.Output("ii1 : int64")
.Output("jj1 : int64")
.Output("vv1 : double")
.Output("ii2 : int64")
.Output("jj2 : int64")
.Output("vv2 : double")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    
        shape_inference::ShapeHandle ii_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &ii_shape));
        shape_inference::ShapeHandle jj_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &jj_shape));
        shape_inference::ShapeHandle vv_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &vv_shape));
        shape_inference::ShapeHandle bd_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 1, &bd_shape));
        shape_inference::ShapeHandle m_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 0, &m_shape));
        shape_inference::ShapeHandle n_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(5), 0, &n_shape));
        shape_inference::ShapeHandle h_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(6), 0, &h_shape));

        c->set_output(0, c->Vector(-1));
        c->set_output(1, c->Vector(-1));
        c->set_output(2, c->Vector(-1));
        c->set_output(3, c->Vector(-1));
        c->set_output(4, c->Vector(-1));
        c->set_output(5, c->Vector(-1));
    return Status::OK();
  });

REGISTER_OP("DirichletBdGrad")

.Input("grad_vv1 : double")
.Input("grad_vv2 : double")
.Input("ii1 : int64")
.Input("jj1 : int64")
.Input("vv1 : double")
.Input("ii2 : int64")
.Input("jj2 : int64")
.Input("vv2 : double")
.Input("ii : int64")
.Input("jj : int64")
.Input("vv : double")
.Input("bd : int32")
.Input("m : int32")
.Input("n : int32")
.Input("h : double")
.Output("grad_ii : int64")
.Output("grad_jj : int64")
.Output("grad_vv : double")
.Output("grad_bd : int32")
.Output("grad_m : int32")
.Output("grad_n : int32")
.Output("grad_h : double");


class DirichletBdOp : public OpKernel {
private:
  
public:
  explicit DirichletBdOp(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(7, context->num_inputs());
    
    
    const Tensor& ii = context->input(0);
    const Tensor& jj = context->input(1);
    const Tensor& vv = context->input(2);
    const Tensor& bd = context->input(3);
    const Tensor& m = context->input(4);
    const Tensor& n = context->input(5);
    const Tensor& h = context->input(6);
    
    
    const TensorShape& ii_shape = ii.shape();
    const TensorShape& jj_shape = jj.shape();
    const TensorShape& vv_shape = vv.shape();
    const TensorShape& bd_shape = bd.shape();
    const TensorShape& m_shape = m.shape();
    const TensorShape& n_shape = n.shape();
    const TensorShape& h_shape = h.shape();
    
    
    DCHECK_EQ(ii_shape.dims(), 1);
    DCHECK_EQ(jj_shape.dims(), 1);
    DCHECK_EQ(vv_shape.dims(), 1);
    DCHECK_EQ(bd_shape.dims(), 1);
    DCHECK_EQ(m_shape.dims(), 0);
    DCHECK_EQ(n_shape.dims(), 0);
    DCHECK_EQ(h_shape.dims(), 0);

    // extra check
        
    
    // get the corresponding Eigen tensors for data access
    
    auto ii_tensor = ii.flat<int64>().data();
    auto jj_tensor = jj.flat<int64>().data();
    auto vv_tensor = vv.flat<double>().data();
    auto bd_tensor = bd.flat<int32>().data();
    auto m_tensor = m.flat<int32>().data();
    auto n_tensor = n.flat<int32>().data();
    auto h_tensor = h.flat<double>().data();

    // implement your forward function here 

    // TODO:
    int N = ii_shape.dim_size(0);
    int bdn = bd_shape.dim_size(0);
    Forward fwd(ii_tensor, jj_tensor, vv_tensor, N ,bd_tensor, bdn,
      *m_tensor, *n_tensor, *h_tensor);
    fwd.fill(context);

  }
};
REGISTER_KERNEL_BUILDER(Name("DirichletBd").Device(DEVICE_CPU), DirichletBdOp);



class DirichletBdGradOp : public OpKernel {
private:
  
public:
  explicit DirichletBdGradOp(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_vv1 = context->input(0);
    const Tensor& grad_vv2 = context->input(1);
    const Tensor& ii1 = context->input(2);
    const Tensor& jj1 = context->input(3);
    const Tensor& vv1 = context->input(4);
    const Tensor& ii2 = context->input(5);
    const Tensor& jj2 = context->input(6);
    const Tensor& vv2 = context->input(7);
    const Tensor& ii = context->input(8);
    const Tensor& jj = context->input(9);
    const Tensor& vv = context->input(10);
    const Tensor& bd = context->input(11);
    const Tensor& m = context->input(12);
    const Tensor& n = context->input(13);
    const Tensor& h = context->input(14);
    
    
    const TensorShape& grad_vv1_shape = grad_vv1.shape();
    const TensorShape& grad_vv2_shape = grad_vv2.shape();
    const TensorShape& ii1_shape = ii1.shape();
    const TensorShape& jj1_shape = jj1.shape();
    const TensorShape& vv1_shape = vv1.shape();
    const TensorShape& ii2_shape = ii2.shape();
    const TensorShape& jj2_shape = jj2.shape();
    const TensorShape& vv2_shape = vv2.shape();
    const TensorShape& ii_shape = ii.shape();
    const TensorShape& jj_shape = jj.shape();
    const TensorShape& vv_shape = vv.shape();
    const TensorShape& bd_shape = bd.shape();
    const TensorShape& m_shape = m.shape();
    const TensorShape& n_shape = n.shape();
    const TensorShape& h_shape = h.shape();
    
    
    DCHECK_EQ(grad_vv1_shape.dims(), 1);
    DCHECK_EQ(grad_vv2_shape.dims(), 1);
    DCHECK_EQ(ii1_shape.dims(), 1);
    DCHECK_EQ(jj1_shape.dims(), 1);
    DCHECK_EQ(vv1_shape.dims(), 1);
    DCHECK_EQ(ii2_shape.dims(), 1);
    DCHECK_EQ(jj2_shape.dims(), 1);
    DCHECK_EQ(vv2_shape.dims(), 1);
    DCHECK_EQ(ii_shape.dims(), 1);
    DCHECK_EQ(jj_shape.dims(), 1);
    DCHECK_EQ(vv_shape.dims(), 1);
    DCHECK_EQ(bd_shape.dims(), 1);
    DCHECK_EQ(m_shape.dims(), 0);
    DCHECK_EQ(n_shape.dims(), 0);
    DCHECK_EQ(h_shape.dims(), 0);

    // extra check
    // int m = Example.dim_size(0);
        
    // create output shape
    
    TensorShape grad_ii_shape(ii_shape);
    TensorShape grad_jj_shape(jj_shape);
    TensorShape grad_vv_shape(vv_shape);
    TensorShape grad_bd_shape(bd_shape);
    TensorShape grad_m_shape(m_shape);
    TensorShape grad_n_shape(n_shape);
    TensorShape grad_h_shape(h_shape);
            
    // create output tensor
    
    Tensor* grad_ii = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_ii_shape, &grad_ii));
    Tensor* grad_jj = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, grad_jj_shape, &grad_jj));
    Tensor* grad_vv = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(2, grad_vv_shape, &grad_vv));
    Tensor* grad_bd = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(3, grad_bd_shape, &grad_bd));
    Tensor* grad_m = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(4, grad_m_shape, &grad_m));
    Tensor* grad_n = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(5, grad_n_shape, &grad_n));
    Tensor* grad_h = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(6, grad_h_shape, &grad_h));
    
    // get the corresponding Eigen tensors for data access
    
    auto ii_tensor = ii.flat<int64>().data();
    auto jj_tensor = jj.flat<int64>().data();
    auto vv_tensor = vv.flat<double>().data();
    auto bd_tensor = bd.flat<int32>().data();
    auto m_tensor = m.flat<int32>().data();
    auto n_tensor = n.flat<int32>().data();
    auto h_tensor = h.flat<double>().data();
    auto grad_vv1_tensor = grad_vv1.flat<double>().data();
    auto grad_vv2_tensor = grad_vv2.flat<double>().data();
    auto ii1_tensor = ii1.flat<int64>().data();
    auto jj1_tensor = jj1.flat<int64>().data();
    auto vv1_tensor = vv1.flat<double>().data();
    auto ii2_tensor = ii2.flat<int64>().data();
    auto jj2_tensor = jj2.flat<int64>().data();
    auto vv2_tensor = vv2.flat<double>().data();
    auto grad_vv_tensor = grad_vv->flat<double>().data();
    auto grad_h_tensor = grad_h->flat<double>().data();   

    // implement your backward function here 
    
    // TODO:
    int N = vv_shape.dim_size(0), bdn = bd_shape.dim_size(0);
    backward(
      grad_vv_tensor, ii_tensor, jj_tensor,
      grad_vv1_tensor, grad_vv2_tensor, N, bd_tensor, bdn,
      *m_tensor, *n_tensor, *h_tensor);
    
  }
};
REGISTER_KERNEL_BUILDER(Name("DirichletBdGrad").Device(DEVICE_CPU), DirichletBdGradOp);


/***************************************************************************************
**********************            GPU Operator            ******************************
***************************************************************************************/


#ifdef USE_GPU
class DirichletBdOpGPU : public OpKernel {
private:
  
public:
  explicit DirichletBdOpGPU(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(7, context->num_inputs());
    
    
    const Tensor& ii = context->input(0);
    const Tensor& jj = context->input(1);
    const Tensor& vv = context->input(2);
    const Tensor& bd = context->input(3);
    const Tensor& m = context->input(4);
    const Tensor& n = context->input(5);
    const Tensor& h = context->input(6);
    
    
    const TensorShape& ii_shape = ii.shape();
    const TensorShape& jj_shape = jj.shape();
    const TensorShape& vv_shape = vv.shape();
    const TensorShape& bd_shape = bd.shape();
    const TensorShape& m_shape = m.shape();
    const TensorShape& n_shape = n.shape();
    const TensorShape& h_shape = h.shape();
    
    
    DCHECK_EQ(ii_shape.dims(), 1);
    DCHECK_EQ(jj_shape.dims(), 1);
    DCHECK_EQ(vv_shape.dims(), 1);
    DCHECK_EQ(bd_shape.dims(), 1);
    DCHECK_EQ(m_shape.dims(), 0);
    DCHECK_EQ(n_shape.dims(), 0);
    DCHECK_EQ(h_shape.dims(), 0);

    // extra check
        
    // create output shape
    
    TensorShape ii1_shape({-1});
    TensorShape jj1_shape({-1});
    TensorShape vv1_shape({-1});
    TensorShape ii2_shape({-1});
    TensorShape jj2_shape({-1});
    TensorShape vv2_shape({-1});
            
    // create output tensor
    
    Tensor* ii1 = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, ii1_shape, &ii1));
    Tensor* jj1 = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, jj1_shape, &jj1));
    Tensor* vv1 = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(2, vv1_shape, &vv1));
    Tensor* ii2 = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(3, ii2_shape, &ii2));
    Tensor* jj2 = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(4, jj2_shape, &jj2));
    Tensor* vv2 = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(5, vv2_shape, &vv2));
    
    // get the corresponding Eigen tensors for data access
    
    auto ii_tensor = ii.flat<int64>().data();
    auto jj_tensor = jj.flat<int64>().data();
    auto vv_tensor = vv.flat<double>().data();
    auto bd_tensor = bd.flat<int32>().data();
    auto m_tensor = m.flat<int32>().data();
    auto n_tensor = n.flat<int32>().data();
    auto h_tensor = h.flat<double>().data();
    auto ii1_tensor = ii1->flat<int64>().data();
    auto jj1_tensor = jj1->flat<int64>().data();
    auto vv1_tensor = vv1->flat<double>().data();
    auto ii2_tensor = ii2->flat<int64>().data();
    auto jj2_tensor = jj2->flat<int64>().data();
    auto vv2_tensor = vv2->flat<double>().data();   

    // implement your forward function here 

    // TODO:

  }
};
REGISTER_KERNEL_BUILDER(Name("DirichletBd").Device(DEVICE_GPU), DirichletBdOpGPU);

class DirichletBdGradOpGPU : public OpKernel {
private:
  
public:
  explicit DirichletBdGradOpGPU(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_vv1 = context->input(0);
    const Tensor& grad_vv2 = context->input(1);
    const Tensor& ii1 = context->input(2);
    const Tensor& jj1 = context->input(3);
    const Tensor& vv1 = context->input(4);
    const Tensor& ii2 = context->input(5);
    const Tensor& jj2 = context->input(6);
    const Tensor& vv2 = context->input(7);
    const Tensor& ii = context->input(8);
    const Tensor& jj = context->input(9);
    const Tensor& vv = context->input(10);
    const Tensor& bd = context->input(11);
    const Tensor& m = context->input(12);
    const Tensor& n = context->input(13);
    const Tensor& h = context->input(14);
    
    
    const TensorShape& grad_vv1_shape = grad_vv1.shape();
    const TensorShape& grad_vv2_shape = grad_vv2.shape();
    const TensorShape& ii1_shape = ii1.shape();
    const TensorShape& jj1_shape = jj1.shape();
    const TensorShape& vv1_shape = vv1.shape();
    const TensorShape& ii2_shape = ii2.shape();
    const TensorShape& jj2_shape = jj2.shape();
    const TensorShape& vv2_shape = vv2.shape();
    const TensorShape& ii_shape = ii.shape();
    const TensorShape& jj_shape = jj.shape();
    const TensorShape& vv_shape = vv.shape();
    const TensorShape& bd_shape = bd.shape();
    const TensorShape& m_shape = m.shape();
    const TensorShape& n_shape = n.shape();
    const TensorShape& h_shape = h.shape();
    
    
    DCHECK_EQ(grad_vv1_shape.dims(), 1);
    DCHECK_EQ(grad_vv2_shape.dims(), 1);
    DCHECK_EQ(ii1_shape.dims(), 1);
    DCHECK_EQ(jj1_shape.dims(), 1);
    DCHECK_EQ(vv1_shape.dims(), 1);
    DCHECK_EQ(ii2_shape.dims(), 1);
    DCHECK_EQ(jj2_shape.dims(), 1);
    DCHECK_EQ(vv2_shape.dims(), 1);
    DCHECK_EQ(ii_shape.dims(), 1);
    DCHECK_EQ(jj_shape.dims(), 1);
    DCHECK_EQ(vv_shape.dims(), 1);
    DCHECK_EQ(bd_shape.dims(), 1);
    DCHECK_EQ(m_shape.dims(), 0);
    DCHECK_EQ(n_shape.dims(), 0);
    DCHECK_EQ(h_shape.dims(), 0);

    // extra check
    // int m = Example.dim_size(0);
        
    // create output shape
    
    TensorShape grad_ii_shape(ii_shape);
    TensorShape grad_jj_shape(jj_shape);
    TensorShape grad_vv_shape(vv_shape);
    TensorShape grad_bd_shape(bd_shape);
    TensorShape grad_m_shape(m_shape);
    TensorShape grad_n_shape(n_shape);
    TensorShape grad_h_shape(h_shape);
            
    // create output tensor
    
    Tensor* grad_ii = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_ii_shape, &grad_ii));
    Tensor* grad_jj = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, grad_jj_shape, &grad_jj));
    Tensor* grad_vv = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(2, grad_vv_shape, &grad_vv));
    Tensor* grad_bd = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(3, grad_bd_shape, &grad_bd));
    Tensor* grad_m = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(4, grad_m_shape, &grad_m));
    Tensor* grad_n = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(5, grad_n_shape, &grad_n));
    Tensor* grad_h = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(6, grad_h_shape, &grad_h));
    
    // get the corresponding Eigen tensors for data access
    
    auto ii_tensor = ii.flat<int64>().data();
    auto jj_tensor = jj.flat<int64>().data();
    auto vv_tensor = vv.flat<double>().data();
    auto bd_tensor = bd.flat<int32>().data();
    auto m_tensor = m.flat<int32>().data();
    auto n_tensor = n.flat<int32>().data();
    auto h_tensor = h.flat<double>().data();
    auto grad_ii1_tensor = grad_ii1.flat<int64>().data();
    auto grad_jj1_tensor = grad_jj1.flat<int64>().data();
    auto grad_vv1_tensor = grad_vv1.flat<double>().data();
    auto grad_ii2_tensor = grad_ii2.flat<int64>().data();
    auto grad_jj2_tensor = grad_jj2.flat<int64>().data();
    auto grad_vv2_tensor = grad_vv2.flat<double>().data();
    auto ii1_tensor = ii1.flat<int64>().data();
    auto jj1_tensor = jj1.flat<int64>().data();
    auto vv1_tensor = vv1.flat<double>().data();
    auto ii2_tensor = ii2.flat<int64>().data();
    auto jj2_tensor = jj2.flat<int64>().data();
    auto vv2_tensor = vv2.flat<double>().data();
    auto grad_ii_tensor = grad_ii->flat<int64>().data();
    auto grad_jj_tensor = grad_jj->flat<int64>().data();
    auto grad_vv_tensor = grad_vv->flat<double>().data();
    auto grad_bd_tensor = grad_bd->flat<int32>().data();
    auto grad_m_tensor = grad_m->flat<int32>().data();
    auto grad_n_tensor = grad_n->flat<int32>().data();
    auto grad_h_tensor = grad_h->flat<double>().data();   

    // implement your backward function here 

    // TODO:
    
  }
};
REGISTER_KERNEL_BUILDER(Name("DirichletBdGrad").Device(DEVICE_GPU), DirichletBdGradOpGPU);

#endif