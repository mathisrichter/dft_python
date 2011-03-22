import DynamicField
import GaussKernel

def main():
    interaction_kernel = GaussKernel.GaussKernel(2)
    interaction_kernel.add_mode(1.0, [0.5, 0.5], [0.0, 0.0])
    interaction_kernel.add_mode(-5.5, [5.5, 5.5], [0.0, 0.0])
    interaction_kernel.calculate()

    field = DynamicField.DynamicField([2,3], interaction_kernel)

    projection = DynamicField.Projection(2, 3, set([0,1]), [2,1])

    projection.set_input_dimensionality(2)
    projection.set_output_dimensionality(3)
    projection.set_input_dimension_sizes([2,3])
    projection.set_output_dimension_sizes([4,3,2])
    projection._incoming_connectables.append(field)

#    for i in range(0, 500):
    field.step()
    projection.step()
    output = projection.get_output()
    print(output)
    print(output.shape)

if __name__ == "__main__":
    main()
