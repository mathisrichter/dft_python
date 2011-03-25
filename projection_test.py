import DynamicField
import Kernel

def main():
    interaction_kernel = Kernel.GaussKernel(1)
    interaction_kernel.add_mode(1.0, [0.5], [0.0])
    interaction_kernel.add_mode(-5.5, [5.5], [0.0])
    interaction_kernel.calculate()

    field_0 = DynamicField.DynamicField([2,3], None)
    field_0.set_name("Field0")
    field_1 = DynamicField.DynamicField([4,3,2], None)
    field_1.set_name("Field1")

    projection = DynamicField.Projection(2, 3, set([0,1]), [2,1])
    projection.set_name("Projection")
    
    DynamicField.connect(field_0, field_1, [projection])

    #projection.set_input_dimensionality(3)
    #projection.set_output_dimensionality(0)
    #projection.set_input_dimension_sizes([3,4,5])
    #projection.set_output_dimension_sizes([1])
    #projection._incoming_connectables.append(field)

#    for i in range(0, 500):
    field_0.step()
    projection.step()
    field_1.step()
    output = projection.get_output()
    print(output)
    print(output.shape)

if __name__ == "__main__":
    main()
