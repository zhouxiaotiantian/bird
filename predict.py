#-----------------------------------------------------------------------#
#   predict.py将单张图片预测、摄像头检测、FPS测试和目录遍历检测等功能
#   整合到了一个py文件中，通过指定mode进行模式的修改。
#-----------------------------------------------------------------------#
import time
import streamlit as st
from streamlit_option_menu import option_menu
import streamlit.components.v1 as html
# import cv2
import requests
import re
import numpy as np
from PIL import Image

from yolo import YOLO

COMMENT_TEMPLATE_MD = """{} - {}
> {}"""

def space(num_lines=1):
    """Adds empty lines to the Streamlit app."""
    for _ in range(num_lines):
        st.write("")

if __name__ == "__main__":
    # Configures the default settings of the page.
    # This must be the first Streamlit command used in your app, and must only be set once.
    st.set_page_config(
        page_title="Hazard Bird Detection",
        page_icon=":baby_chick::baby_chick:",
        layout="wide",
        initial_sidebar_state="expanded"
    )    
    yolo = YOLO()
    #----------------------------------------------------------------------------------------------------------#
    #   mode用于指定测试的模式：
    #   'predict'表示单张图片预测，如果想对预测过程进行修改，如保存图片，截取对象等，可以先看下方详细的注释
    #   'video'表示视频检测，可调用摄像头或者视频进行检测，详情查看下方注释。
    #   'fps'表示测试fps，使用的图片是img里面的street.jpg，详情查看下方注释。
    #   'dir_predict'表示遍历文件夹进行检测并保存。默认遍历img文件夹，保存img_out文件夹，详情查看下方注释。
    #----------------------------------------------------------------------------------------------------------#
    mode = "predict"
    #-------------------------------------------------------------------------#
    #   crop指定了是否在单张图片预测后对目标进行截取
    #   crop仅在mode='predict'时有效
    #-------------------------------------------------------------------------#
    crop            = False
    #----------------------------------------------------------------------------------------------------------#
    #   video_path用于指定视频的路径，当video_path=0时表示检测摄像头
    #   想要检测视频，则设置如video_path = "xxx.mp4"即可，代表读取出根目录下的xxx.mp4文件。
    #   video_save_path表示视频保存的路径，当video_save_path=""时表示不保存
    #   想要保存视频，则设置如video_save_path = "yyy.mp4"即可，代表保存为根目录下的yyy.mp4文件。
    #   video_fps用于保存的视频的fps
    #   video_path、video_save_path和video_fps仅在mode='video'时有效
    #   保存视频时需要ctrl+c退出或者运行到最后一帧才会完成完整的保存步骤。
    #----------------------------------------------------------------------------------------------------------#
    video_path      = 0
    video_save_path = ""
    video_fps       = 25.0
    #-------------------------------------------------------------------------#
    #   test_interval用于指定测量fps的时候，图片检测的次数
    #   理论上test_interval越大，fps越准确。
    #-------------------------------------------------------------------------#
    test_interval   = 100
    #-------------------------------------------------------------------------#
    #   dir_origin_path指定了用于检测的图片的文件夹路径
    #   dir_save_path指定了检测完图片的保存路径
    #   dir_origin_path和dir_save_path仅在mode='dir_predict'时有效
    #-------------------------------------------------------------------------#
    dir_origin_path = "img/"
    dir_save_path   = "img_out/"

    if mode == "predict":
        # '''
        # 1、如果想要进行检测完的图片的保存，利用r_image.save("img.jpg")即可保存，直接在predict.py里进行修改即可。 
        # 2、如果想要获得预测框的坐标，可以进入yolo.detect_image函数，在绘图部分读取top，left，bottom，right这四个值。
        # 3、如果想要利用预测框截取下目标，可以进入yolo.detect_image函数，在绘图部分利用获取到的top，left，bottom，right这四个值
        # 在原图上利用矩阵的方式进行截取。
        # 4、如果想要在预测图上写额外的字，比如检测到的特定目标的数量，可以进入yolo.detect_image函数，在绘图部分对predicted_class进行判断，
        # 比如判断if predicted_class == 'car': 即可判断当前目标是否为车，然后记录数量即可。利用draw.text即可写字。
        # '''

        with st.sidebar:
            choose = option_menu("甄羽Streamlit", ["拍照识鸟", "图片/音乐/视频", "数据可视化", "地图分布", "其他应用"],
                                icons=['camera-fill', 'file-earmark-music', 'bar-chart', 'brightness-high'], # 对应的小图标，不用改
                                menu_icon="broadcast", default_index=0)
        if choose == "拍照识鸟":
            selecte0 = option_menu(None, ["智能识别", "涉鸟隐患"],
                icons=['card-image', 'cloud-upload'],
                menu_icon="cast", default_index=0, orientation="horizontal")
            if selecte0 == "智能识别":
                ####### 
                st.title(':baby_chick:拍照识鸟\n你好 :sunglasses:') #网页上的文本
                st.info('为了处理突发性输电线路渉鸟故障，针对性地加装防鸟措施，:balloon:甄羽可为您识别涉鸟故障危害鸟种，以便为运维人员提供正确识鸟的工具。') #加载图片
                img = st.file_uploader('图片加载处') #加载图片
                if img:
                    try:
                        image = Image.open(img)
                    except:
                        print('Open Error! Try again!')
                    else:# 使用 else 包裹的代码，只有当 try 块没有捕获到任何异常时，才会得到执行
                        r_image = yolo.detect_image(image, crop = crop)
                        st.balloons()
                        st.title('您选择的图片:')
                        st.image(r_image)
                        # r_image.show()
                else:
                    st.title(":exclamation:您还未选择图片")
                    # st.caption("buluuuuuuuuuuuu")
                ######  
            elif selecte0 == "涉鸟隐患":
                st.title("涉鸟隐患")
       

        elif choose == "图片/音乐/视频":
            selecte1 = option_menu(None, ["图片", "音乐", "视频"],
                icons=['house', 'cloud-upload', "list-task"],
                menu_icon="cast", default_index=0, orientation="horizontal")
            if selecte1 == "图片":
                st.title("随便放张图")
                st.image("./photo/1.jpg")
            elif selecte1 == "音乐":
                # st.audio("./音乐/music.mp3")
                st.title("音乐")
            elif selecte1 == "视频":
                # st.video("./视频/地震.mp4")
                st.title("视频")

        elif choose == "数据可视化":
            selecte2 = option_menu(None, ["Echarts", "Plotly", "Streamlit-apex-charts"],
                                icons=['house', 'cloud-upload', "list-task"],
                                menu_icon="cast", default_index=0, orientation="horizontal")
            if selecte2 == "Echarts":
                html.iframe("https://mp.weixin.qq.com/s/5VDGsnpgx8iF90aF7p1yMg")

            elif selecte2 == "Plotly":
                html.iframe("https://mp.weixin.qq.com/s/ckcDXhoRmxlxswOviQUbFg")

            elif selecte2 == "Streamlit-apex-charts":
                st.components.v1.iframe("https://mp.weixin.qq.com/s/Sm3UifwoxVKTsMD-rsyovA")


        elif choose == "地理":
            selecte4 = option_menu(None, ["地震数据", "KML", "Mapinfo TAB"],
                                icons=['house', 'cloud-upload', 'cloud-upload'],
                                menu_icon="cast", default_index=0, orientation="horizontal")

            if selecte4 == "地震数据":
                html.iframe("https://mp.weixin.qq.com/s/HwYQXotuyZAtecOY6SBYKw")

            elif selecte4 == "KML":
                html.iframe("https://mp.weixin.qq.com/s/-z3dLVE-K0ejB6Sye0EOhg")

            elif selecte4 == "Mapinfo TAB":
                html.iframe("https://mp.weixin.qq.com/s/kP731l40Rf61CTWfyqbQmg")


        elif choose == "其他应用":
            st.title("1")

        ##############################################
        # 侧边栏
        # dtype_file_structure_mapping = { #左边菜单栏Data Portion Type的4个选项
        #     '栏目一：拍照识鸟': 'Identify Bird',
        #     '栏目二：xxxx': 'Article Recommendation',
        #     '栏目三：xxxx': 'Recognition Exercise',
        # } 
        # data_split_names = list(dtype_file_structure_mapping.keys())#暂时不用    
        # dataset_type = st.sidebar.selectbox("导航1", data_split_names)
        # image_files_subset = dtype_file_structure_mapping[dataset_type] 
        # if image_files_subset == 'Identify Bird':
        #     st.title('栏目一\n你好 :sunglasses:【Version in 2022.5.23】') #网页上的文本
        #     instructions = """
        #         为了处理突发性输电线路渉鸟故障，针对性地加装防鸟措施，:balloon:甄羽可为您识别涉鸟故障
        #         危害鸟种，以便为运维人员提供正确识鸟的工具。(LXX_Jo)\n
        #         """
        #     st.write(instructions)
        #     st.subheader(':information_desk_person:拍照识鸟：')

        #     # img = input('Input image filename:')
        #     img = st.file_uploader('图片加载处') #加载图片
        #     if img:
        #         try:
        #             image = Image.open(img)
        #         except:
        #             print('Open Error! Try again!')
        #         else:# 使用 else 包裹的代码，只有当 try 块没有捕获到任何异常时，才会得到执行
        #             r_image = yolo.detect_image(image, crop = crop)
        #             st.image(r_image)
        #             # r_image.show()
        #     else:
        #         st.title(":balloon:You have not selected a picture :")
        #         st.caption("buluuuuuuuuuuuu")

        # elif image_files_subset == 'Article Recommendation':
        #     st.title('栏目二') 

        # elif image_files_subset == 'Recognition Exercise':
        #     st.title('栏目三') 

        # selected_species = st.sidebar.selectbox("Bird Type", types_of_birds)
        # available_images = load_list_of_images_available(
        #     all_image_files, image_files_subset, selected_species.upper())
        # image_name = st.sidebar.selectbox("Image Name", available_images)

    # elif mode == "video":
    #     capture = cv2.VideoCapture(video_path)
    #     if video_save_path!="":
    #         fourcc  = cv2.VideoWriter_fourcc(*'XVID')
    #         size    = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    #         out     = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)

    #     fps = 0.0
    #     while(True):
    #         t1 = time.time()
    #         # 读取某一帧
    #         ref, frame = capture.read()
    #         if not ref:
    #             break
    #         # 格式转变，BGRtoRGB
    #         frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    #         # 转变成Image
    #         frame = Image.fromarray(np.uint8(frame))
    #         # 进行检测
    #         frame = np.array(yolo.detect_image(frame))
    #         # RGBtoBGR满足opencv显示格式
    #         frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
            
    #         fps  = ( fps + (1./(time.time()-t1)) ) / 2
    #         print("fps= %.2f"%(fps))
    #         frame = cv2.putText(frame, "fps= %.2f"%(fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
    #         cv2.imshow("video",frame)
    #         c= cv2.waitKey(1) & 0xff 
    #         if video_save_path!="":
    #             out.write(frame)

    #         if c==27:
    #             capture.release()
    #             break

    #     print("Video Detection Done!")
    #     capture.release()
    #     if video_save_path!="":
    #         print("Save processed video to the path :" + video_save_path)
    #         out.release()
    #     cv2.destroyAllWindows()

    # elif mode == "fps":
    #     img = Image.open('img/street.jpg')
    #     tact_time = yolo.get_FPS(img, test_interval)
    #     print(str(tact_time) + ' seconds, ' + str(1/tact_time) + 'FPS, @batch_size 1')

    # elif mode == "dir_predict":
    #     import os

    #     from tqdm import tqdm

    #     img_names = os.listdir(dir_origin_path)
    #     for img_name in tqdm(img_names):
    #         if img_name.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
    #             image_path  = os.path.join(dir_origin_path, img_name)
    #             image       = Image.open(image_path)
    #             r_image     = yolo.detect_image(image)
    #             if not os.path.exists(dir_save_path):
    #                 os.makedirs(dir_save_path)
    #             r_image.save(os.path.join(dir_save_path, img_name.replace(".jpg", ".png")), quality=95, subsampling=0)
                
    else:
        raise AssertionError("Please specify the correct mode: 'predict', 'video', 'fps' or 'dir_predict'.")
