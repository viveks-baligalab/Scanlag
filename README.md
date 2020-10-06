# Scanlag
- This is a program to identify hetergeneity in growth.
- Requires Images folder with with time lapse images aquired on a flatbed documents scanner.
##refer to examples files for format##
    Step 1: Crop individual experiments
            >> crop_images_in_folder(folder_with_images,out_folder_for_cropped_images,plate_location_list,name_list,time0)
            ## folder for cropped images should be created before hand
            ## plate_location_list is a list of format :
                    ### plate_location_list = np.array([[x,y,550],....,[x,y,550]])
                    ### where x,y is the location of centers of plates obtained from imageJ
            ## name_list is the list if names to be given for the plates; should be in same order as the plate_location_list
            ## time0 is the time on the first image.
    Step 2: Calculate growth
            >> growth = get_growth(folder,cell_types,annotation_time_list,size)
            ## size can be 500 to 1500, should be decided based on the annotation
            ## annotation_time_list is the list of times for the experiments which needs to be annotated; this is obtained by checking through crop image folder to find the time for a given experiment where the colonies are distinct and visible.
    Step 3: Normalize to the spot
            >> growth_normalized_to_spots = normalize_to_spots(growth)
    Step 4: Save table
            ## save growth data to csv
            >> growth_normalized_to_spots = normalize_to_spots.to_csv(file_name)
    Step 5: Plot spot intensity vs time
            >> plot_time_vs_growth(growth_normalized_to_spots,"Growth_normalized",name_list,colors)
            ## identify spot intensity to be used as threshold
    Step 6: >> toa = get_time_of_appearance(growth_normalized_to_spots,pixel_threshold)
            ## threshold is ones identified in earlier step
    Step 7: Save TOA file
            >> toa.to_csv(file_name)
