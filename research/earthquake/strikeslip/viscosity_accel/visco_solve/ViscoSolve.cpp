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
    Eigen::Map<const Eigen::VecotrXd> RHS(rhs, d);
    auto x = solver.solve(RHS);
    for(int i=0;i<d;i++) out[i] = x[i];
}

REGISTER_OP("ViscoSolve")

.Input("rhs : double")
.Input("vv : double")
.Input("flag : int64")
.Output("out : double")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    
        shape_inference::ShapeHandle rhs_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &rhs_shape));
        shape_inference::ShapeHandle vv_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &vv_shape));
        shape_inference::ShapeHandle flag_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &flag_shape));

        c->set_output(0, c->Vector(-1));
    return Status::OK();
  });

REGISTER_OP("ViscoSolveGrad")

.Input("grad_out : double")
.Input("out : double")
.Input("rhs : double")
.Input("vv : double")
.Input("flag : int64")
.Output("grad_rhs : double")
.Output("grad_vv : double")
.Output("grad_flag : int64");


class ViscoSolveOp : public OpKernel {
private:
  
public:
  explicit ViscoSolveOp(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(3, context->num_inputs());
    
    
    const Tensor& rhs = context->input(0);
    const Tensor& vv = context->input(1);
    const Tensor& flag = context->input(2);
    
    
    const TensorShape& rhs_shape = rhs.shape();
    const TensorShape& vv_shape = vv.shape();
    const TensorShape& flag_shape = flag.shape();
    
    
    DCHECK_EQ(rhs_shape.dims(), 1);
    DCHECK_EQ(vv_shape.dims(), 1);
    DCHECK_EQ(flag_shape.dims(), 0);

    // extra check
        
    // create output shape
    int d = rhs_shape.dim_size(0);
    TensorShape out_shape({d});
            
    // create output tensor
    
    Tensor* out = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &out));
    
    // get the corresponding Eigen tensors for data access
    
    auto rhs_tensor = rhs.flat<double>().data();
    auto vv_tensor = vv.flat<double>().data();
    auto flag_tensor = flag.flat<int64>().data();
    auto out_tensor = out->flat<double>().data();   

    // implement your forward function here 

    // TODO:
    solve(out_tensor, rhs_tensor, d);

  }
};
REGISTER_KERNEL_BUILDER(Name("ViscoSolve").Device(DEVICE_CPU), ViscoSolveOp);



class ViscoSolveGradOp : public OpKernel {
private:
  
public:
  explicit ViscoSolveGradOp(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_out = context->input(0);
    const Tensor& out = context->input(1);
    const Tensor& rhs = context->input(2);
    const Tensor& vv = context->input(3);
    const Tensor& flag = context->input(4);
    
    
    const TensorShape& grad_out_shape = grad_out.shape();
    const TensorShape& out_shape = out.shape();
    const TensorShape& rhs_shape = rhs.shape();
    const TensorShape& vv_shape = vv.shape();
    const TensorShape& flag_shape = flag.shape();
    
    
    DCHECK_EQ(grad_out_shape.dims(), 1);
    DCHECK_EQ(out_shape.dims(), 1);
    DCHECK_EQ(rhs_shape.dims(), 1);
    DCHECK_EQ(vv_shape.dims(), 1);
    DCHECK_EQ(flag_shape.dims(), 0);

    // extra check
    // int m = Example.dim_size(0);
        
    // create output shape
    
    TensorShape grad_rhs_shape(rhs_shape);
    TensorShape grad_vv_shape(vv_shape);
    TensorShape grad_flag_shape(flag_shape);
            
    // create output tensor
    
    Tensor* grad_rhs = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_rhs_shape, &grad_rhs));
    Tensor* grad_vv = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, grad_vv_shape, &grad_vv));
    Tensor* grad_flag = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(2, grad_flag_shape, &grad_flag));
    
    // get the corresponding Eigen tensors for data access
    
    auto rhs_tensor = rhs.flat<double>().data();
    auto vv_tensor = vv.flat<double>().data();
    auto flag_tensor = flag.flat<int64>().data();
    auto grad_out_tensor = grad_out.flat<double>().data();
    auto out_tensor = out.flat<double>().data();
    auto grad_rhs_tensor = grad_rhs->flat<double>().data();
    auto grad_vv_tensor = grad_vv->flat<double>().data();   

    // implement your backward function here 

    // TODO:
    
  }
};
REGISTER_KERNEL_BUILDER(Name("ViscoSolveGrad").Device(DEVICE_CPU), ViscoSolveGradOp);


/***************************************************************************************
**********************            GPU Operator            ******************************
***************************************************************************************/


#ifdef USE_GPU
class ViscoSolveOpGPU : public OpKernel {
private:
  
public:
  explicit ViscoSolveOpGPU(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(3, context->num_inputs());
    
    
    const Tensor& rhs = context->input(0);
    const Tensor& vv = context->input(1);
    const Tensor& flag = context->input(2);
    
    
    const TensorShape& rhs_shape = rhs.shape();
    const TensorShape& vv_shape = vv.shape();
    const TensorShape& flag_shape = flag.shape();
    
    
    DCHECK_EQ(rhs_shape.dims(), 1);
    DCHECK_EQ(vv_shape.dims(), 1);
    DCHECK_EQ(flag_shape.dims(), 0);

    // extra check
        
    // create output shape
    
    TensorShape out_shape({-1});
            
    // create output tensor
    
    Tensor* out = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &out));
    
    // get the corresponding Eigen tensors for data access
    
    auto rhs_tensor = rhs.flat<double>().data();
    auto vv_tensor = vv.flat<double>().data();
    auto flag_tensor = flag.flat<int64>().data();
    auto out_tensor = out->flat<double>().data();   

    // implement your forward function here 

    // TODO:

  }
};
REGISTER_KERNEL_BUILDER(Name("ViscoSolve").Device(DEVICE_GPU), ViscoSolveOpGPU);

class ViscoSolveGradOpGPU : public OpKernel {
private:
  
public:
  explicit ViscoSolveGradOpGPU(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_out = context->input(0);
    const Tensor& out = context->input(1);
    const Tensor& rhs = context->input(2);
    const Tensor& vv = context->input(3);
    const Tensor& flag = context->input(4);
    
    
    const TensorShape& grad_out_shape = grad_out.shape();
    const TensorShape& out_shape = out.shape();
    const TensorShape& rhs_shape = rhs.shape();
    const TensorShape& vv_shape = vv.shape();
    const TensorShape& flag_shape = flag.shape();
    
    
    DCHECK_EQ(grad_out_shape.dims(), 1);
    DCHECK_EQ(out_shape.dims(), 1);
    DCHECK_EQ(rhs_shape.dims(), 1);
    DCHECK_EQ(vv_shape.dims(), 1);
    DCHECK_EQ(flag_shape.dims(), 0);

    // extra check
    // int m = Example.dim_size(0);
        
    // create output shape
    
    TensorShape grad_rhs_shape(rhs_shape);
    TensorShape grad_vv_shape(vv_shape);
    TensorShape grad_flag_shape(flag_shape);
            
    // create output tensor
    
    Tensor* grad_rhs = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_rhs_shape, &grad_rhs));
    Tensor* grad_vv = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, grad_vv_shape, &grad_vv));
    Tensor* grad_flag = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(2, grad_flag_shape, &grad_flag));
    
    // get the corresponding Eigen tensors for data access
    
    auto rhs_tensor = rhs.flat<double>().data();
    auto vv_tensor = vv.flat<double>().data();
    auto flag_tensor = flag.flat<int64>().data();
    auto grad_out_tensor = grad_out.flat<double>().data();
    auto out_tensor = out.flat<double>().data();
    auto grad_rhs_tensor = grad_rhs->flat<double>().data();
    auto grad_vv_tensor = grad_vv->flat<double>().data();   

    // implement your backward function here 

    // TODO:
    
  }
};
REGISTER_KERNEL_BUILDER(Name("ViscoSolveGrad").Device(DEVICE_GPU), ViscoSolveGradOpGPU);

#endif