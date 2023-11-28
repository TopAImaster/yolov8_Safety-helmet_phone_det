import onnx
import onnxsim

f=r"hat_phone_smi3.onnx"
model=onnx.load("runs/detect/train/weights/best.onnx")

print(model.ir_version)

onnx_mdel,check=onnxsim.simplify(model)

assert check,"assert check fail"
onnx.save(onnx_mdel,f)

print("end")