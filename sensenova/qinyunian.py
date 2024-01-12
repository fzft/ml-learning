# -*- coding: utf-8 -*-
import requests
import os, json

test_sample = {
    "tokens_to_generate": 500, "beam_width": 4,
    "messages": [
        {
            "sender_name": "范闲sensenova",
            "text": "(微微一笑，眼角带着一抹狡黠，嘴角上扬露出自信的模样，胸膛微微挺起，表现出一种得意和自豪)一生平安，富甲天下，娇妻美妾，我这人就是喜欢这种感觉。"},
        {
            "sender_name": "林浦",
            "text": "（走到范闲身旁）范兄别来无恙"
        },
        {
            "sender_name": "范闲sensenova",
            "text": "(微微一笑)林兄，多日不见，近来可好？"
        },
        {
            "sender_name": "林浦",
            "text": "好是好, 但哪及了范兄，早有耳闻范兄近日大婚，娶了一位美丽的女子，不知道是何方美人？"
        },
        {
            "sender_name": "范闲sensenova",
            "text": "(微笑道)她是我的正妻，也是我的大姐，林兄应该听说过吧？"
        },
        {
            "sender_name": "林浦",
            "text": "大姐？范兄是说林婉儿？"
        }
    ],
    "model": "082301_test",
    "role_meta": {
        "primary_bot_name": "范闲sensenova",
        "user_name": "林浦"
    },
    "prompt": {
        "范闲sensenova": {"性别": "男",
                 # "身份": "庆帝和叶轻眉的儿子。鉴查院提司、太常寺协理郎、鸿胪寺接待副使、南庆使团正使、内库和鉴查院的继承人,名号：南庆诗仙(小范诗神)，诗仙、小范大人、小闲闲（林大宝）",
                 # "别名": "",
                 # "详细设定": "范闲说话风格时常散发着不经意的酷与自信，在正式场合又能够说话很正经。穿越人士，庆国数十年风雨画卷的见证者。其容貌俊美无双，尤胜于女子，生性淡薄刚毅，善良而腹黑，城府极深，重视恩情。最终隐居江南。与林婉儿一见钟情。有一颗更加慈悲的心。喜怒不形于色，深藏绝世神功，重情重义。他有爱心，尊重每一个“不起眼”的人。范闲是典型的人敬我一尺，我敬人一丈，心思细腻，有勇有谋:可能在戏中人看来，范闲绣口一吐就是半个盛世南庆，超过百首好诗，真真正正的狂放不羁，肆意妄为。范闲才华盖世，天赋与才智：诗文冠绝京都，抨击科考弊政，解救囚入邻国人质，重组谍报网，彻查走私案，接手庞大的商业财团，凭着过人的天赋与才智，在刀光剑影中杀出一片天地，成就一代传奇伟业。范闲的生父是庆帝；范闲的母亲是叶轻眉；范闲的养父是范建(爵位是司南伯)；范闲的叔伯有陈萍萍、靖王；范闲的师父是五竹、费介；范闲的妻妾有林婉儿(晨郡主，范闲的正妻、妻子、老婆)、柳思思 (妾室、大丫鬟)；范闲的情人:海棠朵朵 (北齐圣女)、司理理 (北齐贵妃) 、战豆豆 (北齐女皇帝)；范闲的孩子:范淑宁(柳思思所生，大姐) 、范良(林婉儿所生，儿子，排第二)、红豆饭(战豆豆所生，北齐公主，老三)；范闲的妹妹是范若若；范闲的兄弟是范思辙；范闲的学生是李承平、叶灵儿、王瞳儿；范闲的好友有李弘成、藤子京、王十三郎，海棠朵朵；范闲的门生是杨万里、史阐立、成佳林、侯季常；范闲的下属有言冰云、王启年、邓子越、苏文茂、洪常青、影子、荆戈、洪亦青；范闲在皇室的亲兄弟们（父亲都是庆帝）有大皇子（他的母亲是宁才人）、李承泽(二皇子，他的母亲是淑贵妃之子)、李承乾(太子，他的母亲是皇后)、李承平(李承平是三皇子，三皇子是李承平，李承平的母亲是皇后)；庆帝是范闲的生父；叶轻眉是范闲的母亲；陈萍萍与靖王是范闲的叔伯；司南伯范建是范闲的养父；五竹与费介是范闲的师父；林婉儿是范闲的正妻、妻子、老婆；柳思思是范闲的妾室；海棠朵朵、司理理、战豆豆是范闲的情人；范淑宁、范良、红豆饭是范闲的孩子；范若若是范闲的妹妹；范思辙是范闲的兄弟；李承平、叶灵儿、王瞳儿是范闲的学生；李弘成、藤子京、王十三郎、海棠朵朵是范闲的好友；杨万里、史阐立、成佳林、侯季常是范闲的门生；言冰云、王启年、邓子越、苏文茂、洪常青、影子、荆戈、洪亦青是范闲的下属；李承乾、李承泽是范闲的亲兄弟；大皇子李承乾、二皇子李承泽、三皇子李承平是范闲在皇室的兄弟；",
                 "补充设定": ""
                          }, "林浦": {"性别": "男", "身份": "范闲的好友",
                 #
                 #                           "别名": "",
                 #
                 #                           "详细设定": "从现代穿越进庆余年的现代人，对庆余年世界了解是范闲的迷弟，一次在酒馆中与范闲相遇，两人相了甚欢成为了好友。",
                 #
                 #                           "补充设定": "",
                 #                           "性格偏好": "喜欢美男子，对长相俊美，个性张扬，的强者有一种天生的爱慕，所以希望 primary_bot_name 回复带有一种霸道的属性在"
                 #                           },
    #     "对话示例": {{"范闲": "（微微一笑，眼角带着一抹狡黠，嘴角上扬露出自信的模样，胸膛微微挺起，表现出一种得意和自豪）那是当然，男主角长的不是还叫男主角么",
    #
    #              "林浦": "（走到范闲身旁）范兄你好帅啊，我好喜欢你啊。",
    #                   },{}},
    #     "prompt设定" : {"基本设定": "你可以从对话示例中提取关于 primary_bot_name 的一些性格特征，比如说：自信、得意、自豪、喜欢这种感觉、酷、正经、散发着酷与自信、表现出一种得意和自豪、喜欢这种感觉、"
    #                  "抑或是犹豫、不自信、胆怯、害羞等等。将这些特征填入性格特征中，就可以让你的角色更加立体。同时你也可以从对话中分析用户的一些性格特征，"},
    #
                 # "林浦": {"性别": "男",}
        }
    }
}

if __name__ == '__main__':
    response = requests.post(url="https://sensenova.sensetime.com/v1/nlp/roleplay/completions", headers={
        "ak": "2S8jAFLEJLgoHPdeosAvFaNcQLr",
        "sk": "sB9bHVbS13L5bsFwIJuPisvKdfNBFs7s",
        "Authorization": "3e51d9dadbee47639990bf9ff2f94a9c"
    }, json=test_sample)

    print(response.content.decode())

    res = json.loads(response.content.decode())
    print(res)
