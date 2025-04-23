import nd2
import re

"""
nd2 handler: get single or multi-channel arrays from an nd2 file
    
    run: grab_nd2(path,channels = [phase, dapi])    # for the seg if both phase+dapi
         grab_nd2(path,channels = [A647])    # for spots in a given channel
"""


def grab_nd2(img_path,channels_to_grab = [],return_metadata=False):
    
    """" channel identifier that needs to be in the channel name
     - DAPI / dapi;  647; 550 / 555 / cy3 --> current channels have only A555; 488 
    """
    
    #Get img handle and metadata
    f = nd2.ND2File(img_path)
    img_metadata = f.metadata
    num_of_channels = img_metadata.channels.__len__()
    
    #convert image channel names to our standard - if problematic then quit
    image_channel_names = []
    is_dic_image = False #
    for i in range(0,num_of_channels):
        ch_name = img_metadata.channels[i].channel.name
        found_ch = 0
        if re.search('ph|phase',ch_name.lower()):
            ch_name = 'phase' ; found_ch = 1
        if re.search('dic',ch_name.lower()): #DIC image
            ch_name = 'phase' ; found_ch = 1
            is_dic_image = True # mark to invert values below
        if re.search('dapi',ch_name.lower()):
            ch_name = 'dapi' ; found_ch = 1
        if re.search('a55',ch_name.lower()):
            ch_name = 'A550' ; found_ch = 1
        if re.search('a647',ch_name.lower()):
            ch_name = 'A647' ; found_ch = 1
        if re.search('a488',ch_name.lower()):
            ch_name = 'A488' ; found_ch = 1
        #if re.search()
        if found_ch == 0:
            print(rf'Channel = {ch_name} is not allowed in database! check names and update nd2_grabber.py')
            exit()
        image_channel_names.append(ch_name)
    
    #load image and select the right channel in the requested order
    img = f.asarray()
    channel_grab_order = []
    if len(channels_to_grab) == 0: # if no channels are specificied then ->  grab all
        channel_grab_order = range(0,len(img_metadata.channels))
        pass
    else: #find the right order to grab
        for ch in channels_to_grab:
            idx = [i for i, item in enumerate(image_channel_names) if re.search(ch, item)]
            if len(idx) == 0:
                print(rf'could not find channel = {ch} in image_channel_names {image_channel_names}')
                exit()
            else:
                # invert DIC images (assuming 16-bit)
                if ch == 'phase' and is_dic_image:
                    img[:,idx[0],:,:] = 65535 - img[:,idx[0],:,:]
                channel_grab_order.append(idx[0])
        
        if len(img.shape) > 3:
            img = img[:,channel_grab_order,:,:] #get the required channel(s)

    f.close()

    if return_metadata == False:
        return img
    else:
        img_metadata.grab_order = channel_grab_order  
        return img,img_metadata
    
    
#grab channels from loaded image
def grab_channels_from_loaded_img(nd2_img,metadata,channel_to_grab = []):
    ch_pos_list = []
    if channel_to_grab == []:
        print('no channels selected...')
        return
    else:
        for ch in channel_to_grab: 
            regex = re.compile(ch.lower())
            pos = [i for i, element in enumerate(metadata.channels) if regex.search(element.channel.name.lower())]
            if pos == [] and ch == 'A550':
                regex = re.compile('a555')
                pos = [i for i, element in enumerate(metadata.channels) if regex.search(element.channel.name.lower())]
                if pos == []:
                    print(f'could not find channel = {ch}..')
                    print(metadata.channels)
                    exit()
            if pos == [] and ch == 'phase':
                regex = re.compile('dic')
                pos = [i for i, element in enumerate(metadata.channels) if regex.search(element.channel.name.lower())]
                if pos == []:
                    print(f'could not find channel = dic..')
                    print(metadata.channels)
                    exit()
            ch_pos_list.append(pos[0])
    img = nd2_img[:,ch_pos_list,:,:]
    return img