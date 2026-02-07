# Imports PIL module
from PIL import Image
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
# open method used to open different extension image file
im1 = Image.open(r"F:\Deepika\Paper_works\Bhasker kantapalli(146203)-paper2(ClassI)\146203bk2\Result\acc_tr1.jpg")
im2 = Image.open(r"F:\Deepika\Paper_works\Bhasker kantapalli(146203)-paper2(ClassI)\146203bk2\Result\tpr_tr1.jpg")
im3 = Image.open(r"F:\Deepika\Paper_works\Bhasker kantapalli(146203)-paper2(ClassI)\146203bk2\Result\tnr_tr1.jpg")
im4 = Image.open(r"F:\Deepika\Paper_works\Bhasker kantapalli(146203)-paper2(ClassI)\146203bk2\Result\acc_k1.jpg")
im5 = Image.open(r"F:\Deepika\Paper_works\Bhasker kantapalli(146203)-paper2(ClassI)\146203bk2\Result\tpr_k1.jpg")
im6 = Image.open(r"F:\Deepika\Paper_works\Bhasker kantapalli(146203)-paper2(ClassI)\146203bk2\Result\tnr_k1.jpg")
im7 = Image.open(r"F:\Deepika\Paper_works\Bhasker kantapalli(146203)-paper2(ClassI)\146203bk2\Result\acc_tr2.jpg")
im8 = Image.open(r"F:\Deepika\Paper_works\Bhasker kantapalli(146203)-paper2(ClassI)\146203bk2\Result\tpr_tr2.jpg")
im9 = Image.open(r"F:\Deepika\Paper_works\Bhasker kantapalli(146203)-paper2(ClassI)\146203bk2\Result\tnr_tr2.jpg")
im10 = Image.open(r"F:\Deepika\Paper_works\Bhasker kantapalli(146203)-paper2(ClassI)\146203bk2\Result\acc_k2.jpg")
im11 = Image.open(r"F:\Deepika\Paper_works\Bhasker kantapalli(146203)-paper2(ClassI)\146203bk2\Result\tpr_k2.jpg")
im12 = Image.open(r"F:\Deepika\Paper_works\Bhasker kantapalli(146203)-paper2(ClassI)\146203bk2\Result\tnr_k2.jpg")



# This method will show image in any image viewer
im1.show()
im2.show()
im3.show()
im4.show()
im5.show()
im6.show()
im7.show()
im8.show()
im9.show()
im10.show()
im11.show()
im12.show()
