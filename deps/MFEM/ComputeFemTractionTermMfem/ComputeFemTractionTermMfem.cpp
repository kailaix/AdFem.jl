#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"
#include<cmath>

// Signatures for GPU kernels here 


using namespace tensorflow;
#include "ComputeFemTractionTermMfem.h"


REGISTER_OP("ComputeFemTractionTermMfem")
.Input("t : double")
.Input("dof : int32")
.Input("bdx : double")
.Input("bdy : double")
.Output("out : double")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    
        shape_inference::ShapeHandle t_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &t_shape));
        shape_inference::ShapeHandle dof_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &dof_shape));
        shape_inference::ShapeHandle bdx_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 2, &bdx_shape));
        shape_inference::ShapeHandle bdy_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 2, &bdy_shape));

        c->set_output(0, c->Vector(-1));
    return Status::OK();
  });

REGISTER_OP("ComputeFemTractionTermMfemGrad")
.Input("grad_out : double")
.Input("out : double")
.Input("t : double")
.Input("dof : int32")
.Input("bdx : double")
.Input("bdy : double")
.Output("grad_t : double")
.Output("grad_dof : int32")
.Output("grad_bdx : double")
.Output("grad_bdy : double");

/*-------------------------------------------------------------------------------------*/

class ComputeFemTractionTermMfemOp : public OpKernel {
private:
  
public:
  explicit ComputeFemTractionTermMfemOp(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(4, context->num_inputs());
    
    
    const Tensor& t = context->input(0);
    const Tensor& dof = context->input(1);
    const Tensor& bdx = context->input(2);
    const Tensor& bdy = context->input(3);
    
    
    const TensorShape& t_shape = t.shape();
    const TensorShape& dof_shape = dof.shape();
    const TensorShape& bdx_shape = bdx.shape();
    const TensorShape& bdy_shape = bdy.shape();
    
    
    DCHECK_EQ(t_shape.dims(), 1);
    DCHECK_EQ(dof_shape.dims(), 1);
    DCHECK_EQ(bdx_shape.dims(), 2);
    DCHECK_EQ(bdy_shape.dims(), 2);

    // extra check
        
    // create output shape
    
    TensorShape out_shape({-1});
            
    // create output tensor
    
    Tensor* out = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &out));
    
    // get the corresponding Eigen tensors for data access
    
    auto t_tensor = t.flat<double>().data();
    auto dof_tensor = dof.flat<int32>().data();
    auto bdx_tensor = bdx.flat<double>().data();
    auto bdy_tensor = bdy.flat<double>().data();
    auto out_tensor = out->flat<double>().data();   

    // implement your forward function here 

    // TODO:

  }
};
REGISTER_KERNEL_BUILDER(Name("ComputeFemTractionTermMfem").Device(DEVICE_CPU), ComputeFemTractionTermMfemOp);



class ComputeFemTractionTermMfemGradOp : public OpKernel {
private:
  
public:
  explicit ComputeFemTractionTermMfemGradOp(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_out = context->input(0);
    const Tensor& out = context->input(1);
    const Tensor& t = context->input(2);
    const Tensor& dof = context->input(3);
    const Tensor& bdx = context->input(4);
    const Tensor& bdy = context->input(5);
    
    
    const TensorShape& grad_out_shape = grad_out.shape();
    const TensorShape& out_shape = out.shape();
    const TensorShape& t_shape = t.shape();
    const TensorShape& dof_shape = dof.shape();
    const TensorShape& bdx_shape = bdx.shape();
    const TensorShape& bdy_shape = bdy.shape();
    
    
    DCHECK_EQ(grad_out_shape.dims(), 1);
    DCHECK_EQ(out_shape.dims(), 1);
    DCHECK_EQ(t_shape.dims(), 1);
    DCHECK_EQ(dof_shape.dims(), 1);
    DCHECK_EQ(bdx_shape.dims(), 2);
    DCHECK_EQ(bdy_shape.dims(), 2);

    // extra check
    // int m = Example.dim_size(0);
        
    // create output shape
    
    TensorShape grad_t_shape(t_shape);
    TensorShape grad_dof_shape(dof_shape);
    TensorShape grad_bdx_shape(bdx_shape);
    TensorShape grad_bdy_shape(bdy_shape);
            
    // create output tensor
    
    Tensor* grad_t = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_t_shape, &grad_t));
    Tensor* grad_dof = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, grad_dof_shape, &grad_dof));
    Tensor* grad_bdx = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(2, grad_bdx_shape, &grad_bdx));
    Tensor* grad_bdy = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(3, grad_bdy_shape, &grad_bdy));
    
    // get the corresponding Eigen tensors for data access
    
    auto t_tensor = t.flat<double>().data();
    auto dof_tensor = dof.flat<int32>().data();
    auto bdx_tensor = bdx.flat<double>().data();
    auto bdy_tensor = bdy.flat<double>().data();
    auto grad_out_tensor = grad_out.flat<double>().data();
    auto out_tensor = out.flat<double>().data();
    auto grad_t_tensor = grad_t->flat<double>().data();
    auto grad_bdx_tensor = grad_bdx->flat<double>().data();
    auto grad_bdy_tensor = grad_bdy->flat<double>().data();   

    // implement your backward function here 

    // TODO:
    
  }
};
REGISTER_KERNEL_BUILDER(Name("ComputeFemTractionTermMfemGrad").Device(DEVICE_CPU), ComputeFemTractionTermMfemGradOp);


/***************************************************************************************
**********************            GPU Operator            ******************************
***************************************************************************************/


#ifdef GOOGLE_CUDA
class ComputeFemTractionTermMfemOpGPU : public OpKernel {
private:
  
public:
  explicit ComputeFemTractionTermMfemOpGPU(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(4, context->num_inputs());
    
    
    const Tensor& t = context->input(0);
    const Tensor& dof = context->input(1);
    const Tensor& bdx = context->input(2);
    const Tensor& bdy = context->input(3);
    
    
    const TensorShape& t_shape = t.shape();
    const TensorShape& dof_shape = dof.shape();
    const TensorShape& bdx_shape = bdx.shape();
    const TensorShape& bdy_shape = bdy.shape();
    
    
    DCHECK_EQ(t_shape.dims(), 1);
    DCHECK_EQ(dof_shape.dims(), 1);
    DCHECK_EQ(bdx_shape.dims(), 2);
    DCHECK_EQ(bdy_shape.dims(), 2);

    // extra check
        
    // create output shape
    
    TensorShape out_shape({-1});
            
    // create output tensor
    
    Tensor* out = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &out));
    
    // get the corresponding Eigen tensors for data access
    
    auto t_tensor = t.flat<double>().data();
    auto dof_tensor = dof.flat<int32>().data();
    auto bdx_tensor = bdx.flat<double>().data();
    auto bdy_tensor = bdy.flat<double>().data();
    auto out_tensor = out->flat<double>().data();   

    // implement your forward function here 

    // TODO:

  }
};
REGISTER_KERNEL_BUILDER(Name("ComputeFemTractionTermMfem").Device(DEVICE_GPU), ComputeFemTractionTermMfemOpGPU);

class ComputeFemTractionTermMfemGradOpGPU : public OpKernel {
private:
  
public:
  explicit ComputeFemTractionTermMfemGradOpGPU(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_out = context->input(0);
    const Tensor& out = context->input(1);
    const Tensor& t = context->input(2);
    const Tensor& dof = context->input(3);
    const Tensor& bdx = context->input(4);
    const Tensor& bdy = context->input(5);
    
    
    const TensorShape& grad_out_shape = grad_out.shape();
    const TensorShape& out_shape = out.shape();
    const TensorShape& t_shape = t.shape();
    const TensorShape& dof_shape = dof.shape();
    const TensorShape& bdx_shape = bdx.shape();
    const TensorShape& bdy_shape = bdy.shape();
    
    
    DCHECK_EQ(grad_out_shape.dims(), 1);
    DCHECK_EQ(out_shape.dims(), 1);
    DCHECK_EQ(t_shape.dims(), 1);
    DCHECK_EQ(dof_shape.dims(), 1);
    DCHECK_EQ(bdx_shape.dims(), 2);
    DCHECK_EQ(bdy_shape.dims(), 2);

    // extra check
    // int m = Example.dim_size(0);
        
    // create output shape
    
    TensorShape grad_t_shape(t_shape);
    TensorShape grad_dof_shape(dof_shape);
    TensorShape grad_bdx_shape(bdx_shape);
    TensorShape grad_bdy_shape(bdy_shape);
            
    // create output tensor
    
    Tensor* grad_t = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_t_shape, &grad_t));
    Tensor* grad_dof = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, grad_dof_shape, &grad_dof));
    Tensor* grad_bdx = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(2, grad_bdx_shape, &grad_bdx));
    Tensor* grad_bdy = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(3, grad_bdy_shape, &grad_bdy));
    
    // get the corresponding Eigen tensors for data access
    
    auto t_tensor = t.flat<double>().data();
    auto dof_tensor = dof.flat<int32>().data();
    auto bdx_tensor = bdx.flat<double>().data();
    auto bdy_tensor = bdy.flat<double>().data();
    auto grad_out_tensor = grad_out.flat<double>().data();
    auto out_tensor = out.flat<double>().data();
    auto grad_t_tensor = grad_t->flat<double>().data();
    auto grad_bdx_tensor = grad_bdx->flat<double>().data();
    auto grad_bdy_tensor = grad_bdy->flat<double>().data();   

    // implement your backward function here 

    // TODO:
    
  }
};
REGISTER_KERNEL_BUILDER(Name("ComputeFemTractionTermMfemGrad").Device(DEVICE_GPU), ComputeFemTractionTermMfemGradOpGPU);

#endif