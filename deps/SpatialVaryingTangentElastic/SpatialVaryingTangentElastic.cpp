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
#include "SpatialVaryingTangentElastic.h"


REGISTER_OP("SpatialVaryingTangentElastic")

.Input("mu : double")
.Input("m : int64")
.Input("n : int64")
.Input("h : double")
.Input("type : int64")
.Output("hmat : double")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    
        shape_inference::ShapeHandle mu_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &mu_shape));
        shape_inference::ShapeHandle m_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &m_shape));
        shape_inference::ShapeHandle n_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &n_shape));
        shape_inference::ShapeHandle h_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &h_shape));
        shape_inference::ShapeHandle type_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 0, &type_shape));

        c->set_output(0, c->MakeShape({-1,2,2}));
    return Status::OK();
  });

REGISTER_OP("SpatialVaryingTangentElasticGrad")

.Input("grad_hmat : double")
.Input("hmat : double")
.Input("mu : double")
.Input("m : int64")
.Input("n : int64")
.Input("h : double")
.Input("type : int64")
.Output("grad_mu : double")
.Output("grad_m : int64")
.Output("grad_n : int64")
.Output("grad_h : double")
.Output("grad_type : int64");


class SpatialVaryingTangentElasticOp : public OpKernel {
private:
  
public:
  explicit SpatialVaryingTangentElasticOp(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(5, context->num_inputs());
    
    
    const Tensor& mu = context->input(0);
    const Tensor& m = context->input(1);
    const Tensor& n = context->input(2);
    const Tensor& h = context->input(3);
    const Tensor& type = context->input(4);
    
    
    const TensorShape& mu_shape = mu.shape();
    const TensorShape& m_shape = m.shape();
    const TensorShape& n_shape = n.shape();
    const TensorShape& h_shape = h.shape();
    const TensorShape& type_shape = type.shape();
    
    
    DCHECK_EQ(mu_shape.dims(), 1);
    DCHECK_EQ(m_shape.dims(), 0);
    DCHECK_EQ(n_shape.dims(), 0);
    DCHECK_EQ(h_shape.dims(), 0);
    DCHECK_EQ(type_shape.dims(), 0);

    // extra check
        
    // create output shape
    auto m_tensor = m.flat<int64>().data();
    auto n_tensor = n.flat<int64>().data();
    TensorShape hmat_shape({4*(*m_tensor)*(*n_tensor),2,2});
            
    // create output tensor
    
    Tensor* hmat = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, hmat_shape, &hmat));
    
    // get the corresponding Eigen tensors for data access
    
    auto mu_tensor = mu.flat<double>().data();
    auto h_tensor = h.flat<double>().data();
    auto type_tensor = type.flat<int64>().data();
    auto hmat_tensor = hmat->flat<double>().data();   

    // implement your forward function here 

    // TODO:
    if (*type_tensor==1){
      forward(hmat_tensor, mu_tensor, *m_tensor, *n_tensor, *h_tensor);
    }
    else if(*type_tensor==2){
      forward2(hmat_tensor, mu_tensor, *m_tensor, *n_tensor, *h_tensor);
    }
    else if (*type_tensor==3){
      forward3(hmat_tensor, mu_tensor, *m_tensor, *n_tensor, *h_tensor);
    }

  }
};
REGISTER_KERNEL_BUILDER(Name("SpatialVaryingTangentElastic").Device(DEVICE_CPU), SpatialVaryingTangentElasticOp);



class SpatialVaryingTangentElasticGradOp : public OpKernel {
private:
  
public:
  explicit SpatialVaryingTangentElasticGradOp(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_hmat = context->input(0);
    const Tensor& hmat = context->input(1);
    const Tensor& mu = context->input(2);
    const Tensor& m = context->input(3);
    const Tensor& n = context->input(4);
    const Tensor& h = context->input(5);
    const Tensor& type = context->input(6);
    
    
    const TensorShape& grad_hmat_shape = grad_hmat.shape();
    const TensorShape& hmat_shape = hmat.shape();
    const TensorShape& mu_shape = mu.shape();
    const TensorShape& m_shape = m.shape();
    const TensorShape& n_shape = n.shape();
    const TensorShape& h_shape = h.shape();
    const TensorShape& type_shape = type.shape();
    
    
    DCHECK_EQ(grad_hmat_shape.dims(), 3);
    DCHECK_EQ(hmat_shape.dims(), 3);
    DCHECK_EQ(mu_shape.dims(), 1);
    DCHECK_EQ(m_shape.dims(), 0);
    DCHECK_EQ(n_shape.dims(), 0);
    DCHECK_EQ(h_shape.dims(), 0);
    DCHECK_EQ(type_shape.dims(), 0);

    // extra check
    // int m = Example.dim_size(0);
        
    // create output shape
    
    TensorShape grad_mu_shape(mu_shape);
    TensorShape grad_m_shape(m_shape);
    TensorShape grad_n_shape(n_shape);
    TensorShape grad_h_shape(h_shape);
    TensorShape grad_type_shape(type_shape);
            
    // create output tensor
    
    Tensor* grad_mu = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_mu_shape, &grad_mu));
    Tensor* grad_m = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, grad_m_shape, &grad_m));
    Tensor* grad_n = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(2, grad_n_shape, &grad_n));
    Tensor* grad_h = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(3, grad_h_shape, &grad_h));
    Tensor* grad_type = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(4, grad_type_shape, &grad_type));
    
    // get the corresponding Eigen tensors for data access
    
    auto mu_tensor = mu.flat<double>().data();
    auto m_tensor = m.flat<int64>().data();
    auto n_tensor = n.flat<int64>().data();
    auto h_tensor = h.flat<double>().data();
    auto type_tensor = type.flat<int64>().data();
    auto grad_hmat_tensor = grad_hmat.flat<double>().data();
    auto hmat_tensor = hmat.flat<double>().data();
    auto grad_mu_tensor = grad_mu->flat<double>().data();
    auto grad_h_tensor = grad_h->flat<double>().data();   

    // implement your backward function here 

    // TODO:

    if (*type_tensor==1){
      backward(
        grad_mu_tensor, grad_hmat_tensor, 
        hmat_tensor, mu_tensor, *m_tensor, *n_tensor, *h_tensor);
    }
    else if (*type_tensor==2){
      backward2(
        grad_mu_tensor, grad_hmat_tensor, 
        hmat_tensor, mu_tensor, *m_tensor, *n_tensor, *h_tensor);
    }
    else if(*type_tensor==3){
      backward3(
        grad_mu_tensor, grad_hmat_tensor, 
        hmat_tensor, mu_tensor, *m_tensor, *n_tensor, *h_tensor);
    }
    
  }
};
REGISTER_KERNEL_BUILDER(Name("SpatialVaryingTangentElasticGrad").Device(DEVICE_CPU), SpatialVaryingTangentElasticGradOp);


/***************************************************************************************
**********************            GPU Operator            ******************************
***************************************************************************************/


#ifdef USE_GPU
class SpatialVaryingTangentElasticOpGPU : public OpKernel {
private:
  
public:
  explicit SpatialVaryingTangentElasticOpGPU(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(5, context->num_inputs());
    
    
    const Tensor& mu = context->input(0);
    const Tensor& m = context->input(1);
    const Tensor& n = context->input(2);
    const Tensor& h = context->input(3);
    const Tensor& type = context->input(4);
    
    
    const TensorShape& mu_shape = mu.shape();
    const TensorShape& m_shape = m.shape();
    const TensorShape& n_shape = n.shape();
    const TensorShape& h_shape = h.shape();
    const TensorShape& type_shape = type.shape();
    
    
    DCHECK_EQ(mu_shape.dims(), 1);
    DCHECK_EQ(m_shape.dims(), 0);
    DCHECK_EQ(n_shape.dims(), 0);
    DCHECK_EQ(h_shape.dims(), 0);
    DCHECK_EQ(type_shape.dims(), 0);

    // extra check
        
    // create output shape
    
    TensorShape hmat_shape({-1,2,2});
            
    // create output tensor
    
    Tensor* hmat = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, hmat_shape, &hmat));
    
    // get the corresponding Eigen tensors for data access
    
    auto mu_tensor = mu.flat<double>().data();
    auto m_tensor = m.flat<int64>().data();
    auto n_tensor = n.flat<int64>().data();
    auto h_tensor = h.flat<double>().data();
    auto type_tensor = type.flat<int64>().data();
    auto hmat_tensor = hmat->flat<double>().data();   

    // implement your forward function here 

    // TODO:

  }
};
REGISTER_KERNEL_BUILDER(Name("SpatialVaryingTangentElastic").Device(DEVICE_GPU), SpatialVaryingTangentElasticOpGPU);

class SpatialVaryingTangentElasticGradOpGPU : public OpKernel {
private:
  
public:
  explicit SpatialVaryingTangentElasticGradOpGPU(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_hmat = context->input(0);
    const Tensor& hmat = context->input(1);
    const Tensor& mu = context->input(2);
    const Tensor& m = context->input(3);
    const Tensor& n = context->input(4);
    const Tensor& h = context->input(5);
    const Tensor& type = context->input(6);
    
    
    const TensorShape& grad_hmat_shape = grad_hmat.shape();
    const TensorShape& hmat_shape = hmat.shape();
    const TensorShape& mu_shape = mu.shape();
    const TensorShape& m_shape = m.shape();
    const TensorShape& n_shape = n.shape();
    const TensorShape& h_shape = h.shape();
    const TensorShape& type_shape = type.shape();
    
    
    DCHECK_EQ(grad_hmat_shape.dims(), 3);
    DCHECK_EQ(hmat_shape.dims(), 3);
    DCHECK_EQ(mu_shape.dims(), 1);
    DCHECK_EQ(m_shape.dims(), 0);
    DCHECK_EQ(n_shape.dims(), 0);
    DCHECK_EQ(h_shape.dims(), 0);
    DCHECK_EQ(type_shape.dims(), 0);

    // extra check
    // int m = Example.dim_size(0);
        
    // create output shape
    
    TensorShape grad_mu_shape(mu_shape);
    TensorShape grad_m_shape(m_shape);
    TensorShape grad_n_shape(n_shape);
    TensorShape grad_h_shape(h_shape);
    TensorShape grad_type_shape(type_shape);
            
    // create output tensor
    
    Tensor* grad_mu = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_mu_shape, &grad_mu));
    Tensor* grad_m = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, grad_m_shape, &grad_m));
    Tensor* grad_n = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(2, grad_n_shape, &grad_n));
    Tensor* grad_h = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(3, grad_h_shape, &grad_h));
    Tensor* grad_type = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(4, grad_type_shape, &grad_type));
    
    // get the corresponding Eigen tensors for data access
    
    auto mu_tensor = mu.flat<double>().data();
    auto m_tensor = m.flat<int64>().data();
    auto n_tensor = n.flat<int64>().data();
    auto h_tensor = h.flat<double>().data();
    auto type_tensor = type.flat<int64>().data();
    auto grad_hmat_tensor = grad_hmat.flat<double>().data();
    auto hmat_tensor = hmat.flat<double>().data();
    auto grad_mu_tensor = grad_mu->flat<double>().data();
    auto grad_h_tensor = grad_h->flat<double>().data();   

    // implement your backward function here 

    // TODO:
    
  }
};
REGISTER_KERNEL_BUILDER(Name("SpatialVaryingTangentElasticGrad").Device(DEVICE_GPU), SpatialVaryingTangentElasticGradOpGPU);

#endif