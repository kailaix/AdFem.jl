#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"
#include<cmath>
using namespace tensorflow;
#include "FemStiffness.h"

#ifdef USE_GPU
#include "tensorflow/core/util/gpu_kernel_helper.h"
namespace tensorflow{
  typedef Eigen::GpuDevice GPUDevice;
  void forwardGPU(const GPUDevice &d);
  void backwardGPU(const GPUDevice &d);
}
#endif


REGISTER_OP("FemStiffness")

.Input("hmat : double")
.Input("m : int32")
.Input("n : int32")
.Input("h : double")
.Output("ii : int64")
.Output("jj : int64")
.Output("vv : double")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    
        shape_inference::ShapeHandle hmat_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &hmat_shape));
        shape_inference::ShapeHandle m_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &m_shape));
        shape_inference::ShapeHandle n_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &n_shape));
        shape_inference::ShapeHandle h_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &h_shape));

        c->set_output(0, c->Vector(-1));
        c->set_output(1, c->Vector(-1));
        c->set_output(2, c->Vector(-1));
    return Status::OK();
  });

REGISTER_OP("FemStiffnessGrad")

.Input("grad_vv : double")
.Input("ii : int64")
.Input("jj : int64")
.Input("vv : double")
.Input("hmat : double")
.Input("m : int32")
.Input("n : int32")
.Input("h : double")
.Output("grad_hmat : double")
.Output("grad_m : int32")
.Output("grad_n : int32")
.Output("grad_h : double");


class FemStiffnessOp : public OpKernel {
private:
  
public:
  explicit FemStiffnessOp(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(4, context->num_inputs());
    
    
    const Tensor& hmat = context->input(0);
    const Tensor& m = context->input(1);
    const Tensor& n = context->input(2);
    const Tensor& h = context->input(3);
    
    
    const TensorShape& hmat_shape = hmat.shape();
    const TensorShape& m_shape = m.shape();
    const TensorShape& n_shape = n.shape();
    const TensorShape& h_shape = h.shape();
    
    
    DCHECK_EQ(hmat_shape.dims(), 2);
    DCHECK_EQ(m_shape.dims(), 0);
    DCHECK_EQ(n_shape.dims(), 0);
    DCHECK_EQ(h_shape.dims(), 0);

    // extra check
        
    
    
    auto hmat_tensor = hmat.flat<double>().data();
    auto m_tensor = m.flat<int32>().data();
    auto n_tensor = n.flat<int32>().data();
    auto h_tensor = h.flat<double>().data(); 

    // implement your forward function here 

    // TODO:
    Forward_FS fwd(hmat_tensor, *m_tensor, *n_tensor, *h_tensor);
    fwd.fill(context);

  }
};
REGISTER_KERNEL_BUILDER(Name("FemStiffness").Device(DEVICE_CPU), FemStiffnessOp);



class FemStiffnessGradOp : public OpKernel {
private:
  
public:
  explicit FemStiffnessGradOp(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_vv = context->input(0);
    const Tensor& ii = context->input(1);
    const Tensor& jj = context->input(2);
    const Tensor& vv = context->input(3);
    const Tensor& hmat = context->input(4);
    const Tensor& m = context->input(5);
    const Tensor& n = context->input(6);
    const Tensor& h = context->input(7);
    
    
    const TensorShape& grad_vv_shape = grad_vv.shape();
    const TensorShape& ii_shape = ii.shape();
    const TensorShape& jj_shape = jj.shape();
    const TensorShape& vv_shape = vv.shape();
    const TensorShape& hmat_shape = hmat.shape();
    const TensorShape& m_shape = m.shape();
    const TensorShape& n_shape = n.shape();
    const TensorShape& h_shape = h.shape();
    
    
    DCHECK_EQ(grad_vv_shape.dims(), 1);
    DCHECK_EQ(ii_shape.dims(), 1);
    DCHECK_EQ(jj_shape.dims(), 1);
    DCHECK_EQ(vv_shape.dims(), 1);
    DCHECK_EQ(hmat_shape.dims(), 2);
    DCHECK_EQ(m_shape.dims(), 0);
    DCHECK_EQ(n_shape.dims(), 0);
    DCHECK_EQ(h_shape.dims(), 0);

    // extra check
    // int m = Example.dim_size(0);
        
    // create output shape
    
    TensorShape grad_hmat_shape(hmat_shape);
    TensorShape grad_m_shape(m_shape);
    TensorShape grad_n_shape(n_shape);
    TensorShape grad_h_shape(h_shape);
            
    // create output tensor
    
    Tensor* grad_hmat = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_hmat_shape, &grad_hmat));
    Tensor* grad_m = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, grad_m_shape, &grad_m));
    Tensor* grad_n = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(2, grad_n_shape, &grad_n));
    Tensor* grad_h = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(3, grad_h_shape, &grad_h));
    
    // // get the corresponding Eigen tensors for data access
    
    auto hmat_tensor = hmat.flat<double>().data();
    auto m_tensor = m.flat<int32>().data();
    auto n_tensor = n.flat<int32>().data();
    auto h_tensor = h.flat<double>().data();
    auto grad_vv_tensor = grad_vv.flat<double>().data();
    auto ii_tensor = ii.flat<int64>().data();
    auto jj_tensor = jj.flat<int64>().data();
    auto vv_tensor = vv.flat<double>().data();
    auto grad_hmat_tensor = grad_hmat->flat<double>().data();

    // implement your backward function here 

    // TODO:
    FS_backward(
      grad_hmat_tensor, grad_vv_tensor,
      *m_tensor, *n_tensor, *h_tensor
    );
    
  }
};
REGISTER_KERNEL_BUILDER(Name("FemStiffnessGrad").Device(DEVICE_CPU), FemStiffnessGradOp);


/***************************************************************************************
**********************            GPU Operator            ******************************
***************************************************************************************/


#ifdef USE_GPU
class FemStiffnessOpGPU : public OpKernel {
private:
  
public:
  explicit FemStiffnessOpGPU(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(4, context->num_inputs());
    
    
    const Tensor& hmat = context->input(0);
    const Tensor& m = context->input(1);
    const Tensor& n = context->input(2);
    const Tensor& h = context->input(3);
    
    
    const TensorShape& hmat_shape = hmat.shape();
    const TensorShape& m_shape = m.shape();
    const TensorShape& n_shape = n.shape();
    const TensorShape& h_shape = h.shape();
    
    
    DCHECK_EQ(hmat_shape.dims(), 2);
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
    
    auto hmat_tensor = hmat.flat<double>().data();
    auto m_tensor = m.flat<int32>().data();
    auto n_tensor = n.flat<int32>().data();
    auto h_tensor = h.flat<double>().data();
    auto ii_tensor = ii->flat<int64>().data();
    auto jj_tensor = jj->flat<int64>().data();
    auto vv_tensor = vv->flat<double>().data();   

    // implement your forward function here 

    // TODO:

  }
};
REGISTER_KERNEL_BUILDER(Name("FemStiffness").Device(DEVICE_GPU), FemStiffnessOpGPU);

class FemStiffnessGradOpGPU : public OpKernel {
private:
  
public:
  explicit FemStiffnessGradOpGPU(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_vv = context->input(0);
    const Tensor& ii = context->input(1);
    const Tensor& jj = context->input(2);
    const Tensor& vv = context->input(3);
    const Tensor& hmat = context->input(4);
    const Tensor& m = context->input(5);
    const Tensor& n = context->input(6);
    const Tensor& h = context->input(7);
    
    
    const TensorShape& grad_vv_shape = grad_vv.shape();
    const TensorShape& ii_shape = ii.shape();
    const TensorShape& jj_shape = jj.shape();
    const TensorShape& vv_shape = vv.shape();
    const TensorShape& hmat_shape = hmat.shape();
    const TensorShape& m_shape = m.shape();
    const TensorShape& n_shape = n.shape();
    const TensorShape& h_shape = h.shape();
    
    
    DCHECK_EQ(grad_vv_shape.dims(), 1);
    DCHECK_EQ(ii_shape.dims(), 1);
    DCHECK_EQ(jj_shape.dims(), 1);
    DCHECK_EQ(vv_shape.dims(), 1);
    DCHECK_EQ(hmat_shape.dims(), 2);
    DCHECK_EQ(m_shape.dims(), 0);
    DCHECK_EQ(n_shape.dims(), 0);
    DCHECK_EQ(h_shape.dims(), 0);

    // extra check
    // int m = Example.dim_size(0);
        
    // create output shape
    
    TensorShape grad_hmat_shape(hmat_shape);
    TensorShape grad_m_shape(m_shape);
    TensorShape grad_n_shape(n_shape);
    TensorShape grad_h_shape(h_shape);
            
    // create output tensor
    
    Tensor* grad_hmat = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_hmat_shape, &grad_hmat));
    Tensor* grad_m = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, grad_m_shape, &grad_m));
    Tensor* grad_n = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(2, grad_n_shape, &grad_n));
    Tensor* grad_h = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(3, grad_h_shape, &grad_h));
    
    // get the corresponding Eigen tensors for data access
    
    auto hmat_tensor = hmat.flat<double>().data();
    auto m_tensor = m.flat<int32>().data();
    auto n_tensor = n.flat<int32>().data();
    auto h_tensor = h.flat<double>().data();
    auto grad_ii_tensor = grad_ii.flat<int64>().data();
    auto grad_jj_tensor = grad_jj.flat<int64>().data();
    auto grad_vv_tensor = grad_vv.flat<double>().data();
    auto ii_tensor = ii.flat<int64>().data();
    auto jj_tensor = jj.flat<int64>().data();
    auto vv_tensor = vv.flat<double>().data();
    auto grad_hmat_tensor = grad_hmat->flat<double>().data();
    auto grad_m_tensor = grad_m->flat<int32>().data();
    auto grad_n_tensor = grad_n->flat<int32>().data();
    auto grad_h_tensor = grad_h->flat<double>().data();   

    // implement your backward function here 

    // TODO:
    
  }
};
REGISTER_KERNEL_BUILDER(Name("FemStiffnessGrad").Device(DEVICE_GPU), FemStiffnessGradOpGPU);

#endif