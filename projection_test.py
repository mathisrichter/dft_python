import DynamicField
import Kernel

def main():
    interaction_kernel_0 = Kernel.GaussKernel(3)
    interaction_kernel_0.add_mode(1.0, [0.5,0.5,0.5], [0.0,0.0,0.0])
    interaction_kernel_0.add_mode(-5.5, [5.5,5.5,5.5], [0.0,0.0,0.0])
    interaction_kernel_0.calculate()

    interaction_kernel_1 = Kernel.GaussKernel(1)
    interaction_kernel_1.add_mode(1.0, [0.5], [0.0])
    interaction_kernel_1.add_mode(-5.5, [5.5], [0.0])
    interaction_kernel_1.calculate()

    field_0 = DynamicField.DynamicField([[5],[2],[10]], [], None)
    field_0.set_boost(20)
    field_1 = DynamicField.DynamicField([[5],[3],[4]], [], None)


    scaler = DynamicField.Scaler()

#    weight = DynamicField.Weight([0., 0.5, 1.0, 0.5, 0.0])
    weight = DynamicField.Weight(5.)

    projection_0 = DynamicField.Projection(3, 0, set([]), [])
    projection_1 = DynamicField.Projection(0, 3, set([]), [])

    processing_steps = [projection_0, scaler, weight, projection_1]
    
    DynamicField.connect(field_0, field_1, processing_steps)


#    for i in range(0, 500):
    field_0.step()
    print("field 0:")
    print(field_0.get_output())
    print(field_0.get_output().shape)
    for processing_step in processing_steps:
        processing_step.step()
        print(processing_step.get_name() + ": ")
        print(processing_step.get_output())
        print(processing_step.get_output().shape)
    field_1.step()
    print("field 1:")
    print(field_1.get_activation())
    print(field_1.get_activation().shape)

if __name__ == "__main__":
    main()
