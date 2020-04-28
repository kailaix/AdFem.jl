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
#include "Common.h"

void solve(double * out, const double * rhs, int d){
    Eigen::Map<const Eigen::VectorXd> RHS(rhs, d);
    auto x = solver.solve(RHS);
    for(int i=0;i<d;i++) out[i] = x[i];
}

REGISTER_OP("PushMatrices")

.Input("ii1 : int64")
.Input("jj1 : int64")
.Input("vv1 : double")
.Input("ii2 : int64")
.Input("jj2 : int64")
.Input("vv2 : double")
.Input("d : int64")
.Output("flag : int64")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    
        shape_inference::ShapeHandle ii1_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &ii1_shape));
        shape_inference::ShapeHandle jj1_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &jj1_shape));
        shape_inference::ShapeHandle vv1_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &vv1_shape));
        shape_inference::ShapeHandle ii2_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 1, &ii2_shape));
        shape_inference::ShapeHandle jj2_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 1, &jj2_shape));
        shape_inference::ShapeHandle vv2_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(5), 1, &vv2_shape));
        shape_inference::ShapeHandle d_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(6), 0, &d_shape));

        c->set_output(0, c->Scalar());
    return Status::OK();
  });

REGISTER_OP("PushMatricesGrad")

.Input("flag : int64")
.Input("ii1 : int64")
.Input("jj1 : int64")
.Input("vv1 : double")
.Input("ii2 : int64")
.Input("jj2 : int64")
.Input("vv2 : double")
.Input("d : int64")
.Output("grad_ii1 : int64")
.Output("grad_jj1 : int64")
.Output("grad_vv1 : double")
.Output("grad_ii2 : int64")
.Output("grad_jj2 : int64")
.Output("grad_vv2 : double")
.Output("grad_d : int64");


class PushMatricesOp : public OpKernel {
private:
  
public:
  explicit PushMatricesOp(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(7, context->num_inputs());
    
    
    const Tensor& ii1 = context->input(0);
    const Tensor& jj1 = context->input(1);
    const Tensor& vv1 = context->input(2);
    const Tensor& ii2 = context->input(3);
    const Tensor& jj2 = context->input(4);
    const Tensor& vv2 = context->input(5);
    const Tensor& d = context->input(6);
    
    
    const TensorShape& ii1_shape = ii1.shape();
    const TensorShape& jj1_shape = jj1.shape();
    const TensorShape& vv1_shape = vv1.shape();
    const TensorShape& ii2_shape = ii2.shape();
    const TensorShape& jj2_shape = jj2.shape();
    const TensorShape& vv2_shape = vv2.shape();
    const TensorShape& d_shape = d.shape();
    
    
    DCHECK_EQ(ii1_shape.dims(), 1);
    DCHECK_EQ(jj1_shape.dims(), 1);
    DCHECK_EQ(vv1_shape.dims(), 1);
    DCHECK_EQ(ii2_shape.dims(), 1);
    DCHECK_EQ(jj2_shape.dims(), 1);
    DCHECK_EQ(vv2_shape.dims(), 1);
    DCHECK_EQ(d_shape.dims(), 0);

    // extra check
        
    // create output shape
    
    TensorShape flag_shape({});
            
    // create output tensor
    
    Tensor* flag = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, flag_shape, &flag));
    
    // get the corresponding Eigen tensors for data access
    
    auto ii1_tensor = ii1.flat<int64>().data();
    auto jj1_tensor = jj1.flat<int64>().data();
    auto vv1_tensor = vv1.flat<double>().data();
    auto ii2_tensor = ii2.flat<int64>().data();
    auto jj2_tensor = jj2.flat<int64>().data();
    auto vv2_tensor = vv2.flat<double>().data();
    auto d_tensor = d.flat<int64>().data();
    auto flag_tensor = flag->flat<int64>().data();   

    // implement your forward function here 

    // TODO:

  }
};
REGISTER_KERNEL_BUILDER(Name("PushMatrices").Device(DEVICE_CPU), PushMatricesOp);



class PushMatricesGradOp : public OpKernel {
private:
  
public:
  explicit PushMatricesGradOp(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& flag = context->input(0);
    const Tensor& ii1 = context->input(1);
    const Tensor& jj1 = context->input(2);
    const Tensor& vv1 = context->input(3);
    const Tensor& ii2 = context->input(4);
    const Tensor& jj2 = context->input(5);
    const Tensor& vv2 = context->input(6);
    const Tensor& d = context->input(7);
    
    
    const TensorShape& flag_shape = flag.shape();
    const TensorShape& ii1_shape = ii1.shape();
    const TensorShape& jj1_shape = jj1.shape();
    const TensorShape& vv1_shape = vv1.shape();
    const TensorShape& ii2_shape = ii2.shape();
    const TensorShape& jj2_shape = jj2.shape();
    const TensorShape& vv2_shape = vv2.shape();
    const TensorShape& d_shape = d.shape();
    
    
    DCHECK_EQ(flag_shape.dims(), 0);
    DCHECK_EQ(ii1_shape.dims(), 1);
    DCHECK_EQ(jj1_shape.dims(), 1);
    DCHECK_EQ(vv1_shape.dims(), 1);
    DCHECK_EQ(ii2_shape.dims(), 1);
    DCHECK_EQ(jj2_shape.dims(), 1);
    DCHECK_EQ(vv2_shape.dims(), 1);
    DCHECK_EQ(d_shape.dims(), 0);

    // extra check
    // int m = Example.dim_size(0);
        
    // create output shape
    
    TensorShape grad_ii1_shape(ii1_shape);
    TensorShape grad_jj1_shape(jj1_shape);
    TensorShape grad_vv1_shape(vv1_shape);
    TensorShape grad_ii2_shape(ii2_shape);
    TensorShape grad_jj2_shape(jj2_shape);
    TensorShape grad_vv2_shape(vv2_shape);
    TensorShape grad_d_shape(d_shape);
            
    // create output tensor
    
    Tensor* grad_ii1 = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_ii1_shape, &grad_ii1));
    Tensor* grad_jj1 = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, grad_jj1_shape, &grad_jj1));
    Tensor* grad_vv1 = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(2, grad_vv1_shape, &grad_vv1));
    Tensor* grad_ii2 = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(3, grad_ii2_shape, &grad_ii2));
    Tensor* grad_jj2 = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(4, grad_jj2_shape, &grad_jj2));
    Tensor* grad_vv2 = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(5, grad_vv2_shape, &grad_vv2));
    Tensor* grad_d = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(6, grad_d_shape, &grad_d));
    
    // get the corresponding Eigen tensors for data access
    
    auto ii1_tensor = ii1.flat<int64>().data();
    auto jj1_tensor = jj1.flat<int64>().data();
    auto vv1_tensor = vv1.flat<double>().data();
    auto ii2_tensor = ii2.flat<int64>().data();
    auto jj2_tensor = jj2.flat<int64>().data();
    auto vv2_tensor = vv2.flat<double>().data();
    auto d_tensor = d.flat<int64>().data();
    auto flag_tensor = flag.flat<int64>().data();
    auto grad_vv1_tensor = grad_vv1->flat<double>().data();
    auto grad_vv2_tensor = grad_vv2->flat<double>().data();   

    // implement your backward function here 

    // TODO:
    
  }
};
REGISTER_KERNEL_BUILDER(Name("PushMatricesGrad").Device(DEVICE_CPU), PushMatricesGradOp);


/***************************************************************************************
**********************            GPU Operator            ******************************
***************************************************************************************/


#ifdef USE_GPU
class PushMatricesOpGPU : public OpKernel {
private:
  
public:
  explicit PushMatricesOpGPU(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(7, context->num_inputs());
    
    
    const Tensor& ii1 = context->input(0);
    const Tensor& jj1 = context->input(1);
    const Tensor& vv1 = context->input(2);
    const Tensor& ii2 = context->input(3);
    const Tensor& jj2 = context->input(4);
    const Tensor& vv2 = context->input(5);
    const Tensor& d = context->input(6);
    
    
    const TensorShape& ii1_shape = ii1.shape();
    const TensorShape& jj1_shape = jj1.shape();
    const TensorShape& vv1_shape = vv1.shape();
    const TensorShape& ii2_shape = ii2.shape();
    const TensorShape& jj2_shape = jj2.shape();
    const TensorShape& vv2_shape = vv2.shape();
    const TensorShape& d_shape = d.shape();
    
    
    DCHECK_EQ(ii1_shape.dims(), 1);
    DCHECK_EQ(jj1_shape.dims(), 1);
    DCHECK_EQ(vv1_shape.dims(), 1);
    DCHECK_EQ(ii2_shape.dims(), 1);
    DCHECK_EQ(jj2_shape.dims(), 1);
    DCHECK_EQ(vv2_shape.dims(), 1);
    DCHECK_EQ(d_shape.dims(), 0);

    // extra check
        
    // create output shape
    
    TensorShape flag_shape({});
            
    // create output tensor
    
    Tensor* flag = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, flag_shape, &flag));
    
    // get the corresponding Eigen tensors for data access
    
    auto ii1_tensor = ii1.flat<int64>().data();
    auto jj1_tensor = jj1.flat<int64>().data();
    auto vv1_tensor = vv1.flat<double>().data();
    auto ii2_tensor = ii2.flat<int64>().data();
    auto jj2_tensor = jj2.flat<int64>().data();
    auto vv2_tensor = vv2.flat<double>().data();
    auto d_tensor = d.flat<int64>().data();
    auto flag_tensor = flag->flat<int64>().data();   

    // implement your forward function here 

    // TODO:

  }
};
REGISTER_KERNEL_BUILDER(Name("PushMatrices").Device(DEVICE_GPU), PushMatricesOpGPU);

class PushMatricesGradOpGPU : public OpKernel {
private:
  
public:
  explicit PushMatricesGradOpGPU(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& flag = context->input(0);
    const Tensor& ii1 = context->input(1);
    const Tensor& jj1 = context->input(2);
    const Tensor& vv1 = context->input(3);
    const Tensor& ii2 = context->input(4);
    const Tensor& jj2 = context->input(5);
    const Tensor& vv2 = context->input(6);
    const Tensor& d = context->input(7);
    
    
    const TensorShape& flag_shape = flag.shape();
    const TensorShape& ii1_shape = ii1.shape();
    const TensorShape& jj1_shape = jj1.shape();
    const TensorShape& vv1_shape = vv1.shape();
    const TensorShape& ii2_shape = ii2.shape();
    const TensorShape& jj2_shape = jj2.shape();
    const TensorShape& vv2_shape = vv2.shape();
    const TensorShape& d_shape = d.shape();
    
    
    DCHECK_EQ(flag_shape.dims(), 0);
    DCHECK_EQ(ii1_shape.dims(), 1);
    DCHECK_EQ(jj1_shape.dims(), 1);
    DCHECK_EQ(vv1_shape.dims(), 1);
    DCHECK_EQ(ii2_shape.dims(), 1);
    DCHECK_EQ(jj2_shape.dims(), 1);
    DCHECK_EQ(vv2_shape.dims(), 1);
    DCHECK_EQ(d_shape.dims(), 0);

    // extra check
    // int m = Example.dim_size(0);
        
    // create output shape
    
    TensorShape grad_ii1_shape(ii1_shape);
    TensorShape grad_jj1_shape(jj1_shape);
    TensorShape grad_vv1_shape(vv1_shape);
    TensorShape grad_ii2_shape(ii2_shape);
    TensorShape grad_jj2_shape(jj2_shape);
    TensorShape grad_vv2_shape(vv2_shape);
    TensorShape grad_d_shape(d_shape);
            
    // create output tensor
    
    Tensor* grad_ii1 = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_ii1_shape, &grad_ii1));
    Tensor* grad_jj1 = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, grad_jj1_shape, &grad_jj1));
    Tensor* grad_vv1 = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(2, grad_vv1_shape, &grad_vv1));
    Tensor* grad_ii2 = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(3, grad_ii2_shape, &grad_ii2));
    Tensor* grad_jj2 = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(4, grad_jj2_shape, &grad_jj2));
    Tensor* grad_vv2 = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(5, grad_vv2_shape, &grad_vv2));
    Tensor* grad_d = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(6, grad_d_shape, &grad_d));
    
    // get the corresponding Eigen tensors for data access
    
    auto ii1_tensor = ii1.flat<int64>().data();
    auto jj1_tensor = jj1.flat<int64>().data();
    auto vv1_tensor = vv1.flat<double>().data();
    auto ii2_tensor = ii2.flat<int64>().data();
    auto jj2_tensor = jj2.flat<int64>().data();
    auto vv2_tensor = vv2.flat<double>().data();
    auto d_tensor = d.flat<int64>().data();
    auto flag_tensor = flag.flat<int64>().data();
    auto grad_vv1_tensor = grad_vv1->flat<double>().data();
    auto grad_vv2_tensor = grad_vv2->flat<double>().data();   

    // implement your backward function here 

    // TODO:
    
  }
};
REGISTER_KERNEL_BUILDER(Name("PushMatricesGrad").Device(DEVICE_GPU), PushMatricesGradOpGPU);

#endif