#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"
#include<cmath>

// Signatures for GPU kernels here 


using namespace tensorflow;
#include "ComputeFemAdvectionMatrixMfem.h"


REGISTER_OP("ComputeFemAdvectionMatrixMfem")
.Input("u : double")
.Input("v : double")
.Output("indices : int64")
.Output("vv : double")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    
        shape_inference::ShapeHandle u_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &u_shape));
        shape_inference::ShapeHandle v_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &v_shape));

        c->set_output(0, c->Matrix(-1,2));
        c->set_output(1, c->Vector(-1));
    return Status::OK();
  });

REGISTER_OP("ComputeFemAdvectionMatrixMfemGrad")
.Input("grad_vv : double")
.Input("indices : int64")
.Input("vv : double")
.Input("u : double")
.Input("v : double")
.Output("grad_u : double")
.Output("grad_v : double");

/*-------------------------------------------------------------------------------------*/

class ComputeFemAdvectionMatrixMfemOp : public OpKernel {
private:
  
public:
  explicit ComputeFemAdvectionMatrixMfemOp(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(2, context->num_inputs());
    
    
    const Tensor& u = context->input(0);
    const Tensor& v = context->input(1);
    
    
    const TensorShape& u_shape = u.shape();
    const TensorShape& v_shape = v.shape();
    
    
    DCHECK_EQ(u_shape.dims(), 1);
    DCHECK_EQ(v_shape.dims(), 1);

    // extra check
        
    // create output shape
    
    TensorShape indices_shape({mmesh.ngauss * mmesh.elem_ndof * mmesh.elem_ndof,2});
    TensorShape vv_shape({mmesh.ngauss * mmesh.elem_ndof * mmesh.elem_ndof});
            
    // create output tensor
    
    Tensor* indices = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, indices_shape, &indices));
    Tensor* vv = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, vv_shape, &vv));
    
    // get the corresponding Eigen tensors for data access
    
    auto u_tensor = u.flat<double>().data();
    auto v_tensor = v.flat<double>().data();
    auto indices_tensor = indices->flat<int64>().data();
    auto vv_tensor = vv->flat<double>().data();   

    // implement your forward function here 

    // TODO:
    MFEM::ComputeFemAdvectionMatrix_forward(indices_tensor, vv_tensor, u_tensor, v_tensor);
  }
};
REGISTER_KERNEL_BUILDER(Name("ComputeFemAdvectionMatrixMfem").Device(DEVICE_CPU), ComputeFemAdvectionMatrixMfemOp);



class ComputeFemAdvectionMatrixMfemGradOp : public OpKernel {
private:
  
public:
  explicit ComputeFemAdvectionMatrixMfemGradOp(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_vv = context->input(0);
    const Tensor& indices = context->input(1);
    const Tensor& vv = context->input(2);
    const Tensor& u = context->input(3);
    const Tensor& v = context->input(4);
    
    
    const TensorShape& grad_vv_shape = grad_vv.shape();
    const TensorShape& indices_shape = indices.shape();
    const TensorShape& vv_shape = vv.shape();
    const TensorShape& u_shape = u.shape();
    const TensorShape& v_shape = v.shape();
    
    
    DCHECK_EQ(grad_vv_shape.dims(), 1);
    DCHECK_EQ(indices_shape.dims(), 2);
    DCHECK_EQ(vv_shape.dims(), 1);
    DCHECK_EQ(u_shape.dims(), 1);
    DCHECK_EQ(v_shape.dims(), 1);

    // extra check
    // int m = Example.dim_size(0);
        
    // create output shape
    
    TensorShape grad_u_shape(u_shape);
    TensorShape grad_v_shape(v_shape);
            
    // create output tensor
    
    Tensor* grad_u = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_u_shape, &grad_u));
    Tensor* grad_v = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, grad_v_shape, &grad_v));
    
    // get the corresponding Eigen tensors for data access
    
    auto u_tensor = u.flat<double>().data();
    auto v_tensor = v.flat<double>().data();
    auto grad_vv_tensor = grad_vv.flat<double>().data();
    auto indices_tensor = indices.flat<int64>().data();
    auto vv_tensor = vv.flat<double>().data();
    auto grad_u_tensor = grad_u->flat<double>().data();
    auto grad_v_tensor = grad_v->flat<double>().data();   

    // implement your backward function here 

    // TODO:
    grad_u->flat<double>().setZero();
    grad_v->flat<double>().setZero();
    MFEM::ComputeFemAdvectionMatrix_backward(grad_u_tensor, grad_v_tensor, grad_vv_tensor);
  }
};
REGISTER_KERNEL_BUILDER(Name("ComputeFemAdvectionMatrixMfemGrad").Device(DEVICE_CPU), ComputeFemAdvectionMatrixMfemGradOp);

