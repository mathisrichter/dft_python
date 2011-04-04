import BehavioralOrganization as BehOrg

def main():

    elem_behavior = BehOrg.ElementaryBehavior(field_dimensionality=2,
                                              field_sizes=[[5],[3]],
                                              field_resolutions=[],
                                              int_node_to_int_field_weight=5.5,
                                              cos_field_to_cos_node_weight=5.5,
                                              cos_node_to_cos_memory_node_weight=5.5,
                                              int_inhibition_weight=-5.5,
                                              reactivating=False)

    for i in range(20):
        elem_behavior.step()


if __name__ == "__main__":
    main()
