from discord.ext import commands
import cv2
import numpy as np
import io
import requests
from fanservice_meter import fsm_ai, DISC_TKN
bot = commands.Bot(command_prefix='?')

ai = fsm_ai()
@bot.command()
async def isthishentai(ctx):
    async with ctx.typing():
        reference = None
        reference = ctx.message.reference
        try:
            m_id = reference.message_id
        except AttributeError:
            await ctx.send("No message was replied to, cannot process nothing")
            return
        m = await ctx.channel.fetch_message(m_id)
        url = None
        try:
            url = m.attachments[0]
        except IndexError:
            await ctx.send("Message referenced has no attached images to analyze")
            return
        print(url)
        image_stream = io.BytesIO()
        data = requests.get(url, stream = True)
        data.raw.decode_content = True
        image_stream.write(data.content)
        image_stream.seek(0)
        file_bytes = np.asarray(bytearray(image_stream.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        cvimg = cv2.cvtColor(cv2.resize(img, ai.img_size), cv2.COLOR_BGR2GRAY) # reduces dimention 
        cvimg = cv2.cvtColor(cvimg, cv2.COLOR_GRAY2BGR) # we need to get it back to 3 channels
        score, label = ai.test_img(cvimg)
        await ctx.send("The AI is %.1f%% sure that the image analyzed is %s" % (score, label)) #TODO: make it call the user a weeb if they send hentai

print('starting bot')
bot.run(DISC_TKN)
