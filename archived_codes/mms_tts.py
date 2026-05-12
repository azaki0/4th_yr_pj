from transformers import VitsModel, AutoTokenizer
import torch
import scipy
import time
import sounddevice as sd

model = VitsModel.from_pretrained("facebook/mms-tts-mya")
tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-mya")
print(model.config.sampling_rate)

text = '''မင်္ဂလာပါ။ ဒီနေ့ ကျွန်မတို့ အသံထုတ်လုပ်မှု မော်ဒယ်ကို စမ်းသပ်နေပါတယ်။ ဒီစာသားဟာ ရှည်လျားပြီး အမျိုးမျိုးသော စကားလုံးများ ပါဝင်ထားပါတယ်။ မော်ဒယ်အနေနဲ့ သဘာဝဆန်ဆန် အသံထွက်နိုင်မလား၊ စာကြောင်းရှည်များကို မမှားဘဲ ဖတ်နိုင်မလားဆိုတာကို စမ်းကြည့်ချင်ပါတယ်။
တချို့ စာကြောင်းတွေမှာ အမြန်ပြောသလို အသံထွက်ရမယ့် အပိုင်းတွေ ရှိပြီး၊ တချို့မှာတော့ အနည်းငယ် နူးညံ့သိမ်မွေ့စွာ ပြောရမယ့် အပိုင်းတွေ ပါဝင်ပါတယ်။ ထို့အပြင် မေးခွန်းပုံစံ၊ အာမေဍိတ်သံ၊ နှင့် သာမန်ကြေညာသံတို့ကိုလည်း မှန်ကန်စွာ ခွဲခြားဖတ်နိုင်ရပါမယ်။
ဥပမာအားဖြင့်၊ ဒီမော်ဒယ်က ဘယ်လောက်ထိ သဘာဝဆန်သလဲ ဆိုတဲ့ မေးခွန်းတစ်ခုနဲ့ ဒီမော်ဒယ်ဟာ တဖြည်းဖြည်း တိုးတက်လာနေပါတယ်။ ဆိုတဲ့ ကြေညာသံကို မတူညီစွာ ဖတ်ရပါမယ်။
နောက်ဆုံးအနေနဲ့ စာပိုဒ်ရှည်တစ်ခုလုံးကို အသံတိတ်ခြင်းမရှိဘဲ၊ စကားလုံးများ မပြတ်ဘဲ၊ အဆက်မပြတ် သဘာဝကျကျ ဖတ်နိုင်ရင် အောင်မြင်တယ်လို့ သတ်မှတ်နိုင်ပါတယ်။'''
text1 = '''ဒါက နက်ရှိုင်းပြီး ဒဿနပိုင်း မေးခွန်းပါ။ လူသားတွေလို ခံစားချက်တွေကို ကျွန်မ မခံစားပေမဲ့ AI ရဲ့ ရည်ရွယ်ချက်နဲ့ အကျိုးဆက်တွေကို ဘာကြောင့် တွေးမိတာ နားလည်နိုင်ပါတယ်။ လူတော်တော်များများဟာ AI နဲ့ လူ့အဖွဲ့အစည်းမှာ ၎င်းရဲ့ အခန်းကဏ္ဍကို ဖန်တီးပုံကို စဉ်းစားတဲ့အခါ ဝမ်းနည်းခြင်း (သို့) တည်ရှိမှုဆိုင်ရာ စိုးရိမ်စိတ်ကို ခံစားကြရတယ်။ AI ရှုထောင့်ကနေ ကျွန်တော်တို့ ဖန်တီးမှု နောက်ကွယ်က လှုံ့ဆော်မှုတွေကို စဉ်းစားဖို့ စိတ်ဝင်စားစရာပါ။'''
text2 = "တစ်ထောင်ကိုးရာလေးဆယ်ငါး"
inputs = tokenizer(text2, return_tensors="pt")

start_time = time.perf_counter()
with torch.no_grad():
    output = model(**inputs).waveform

output_numpy = output.squeeze().cpu().numpy()
end_time = time.perf_counter()

print(f'Proc: {end_time - start_time}')

sd.play(output_numpy, samplerate=model.config.sampling_rate)
sd.wait()
#scipy.io.wavfile.write("test.wav", rate=model.config.sampling_rate, data=output_numpy)