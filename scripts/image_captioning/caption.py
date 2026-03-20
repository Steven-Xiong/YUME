from openai import AzureOpenAI

# client = AzureOpenAI(
#     api_key="fCTG9hlWLCmMr7ZfEiWhezW0rCPYo274_GPT_AK",
#     api_version="2024-02-01",
#     azure_endpoint="https://aidp-i18ntt-sg.byteintl.net/api/modelhub/online/v2/crawl",
#     default_headers={"X-TT-LOGID": "1234567890"}
# )

# response = client.chat.completions.create(
#     model="gpt-5.2-2025-12-11",
#     messages=[
#         {
#             "role": "user",
#             "content": [
#                 {
#                     "type": "text",
#                     "text": "What is the result of 1+1?"
#                 }
#             ]
#         }
#     ],
#     max_tokens=500,
#     stream=False
# )

# print(response.model_dump_json(indent=2))




#  （注意：办公网络须将域名改为 https://aidp-i18ntt-sg.tiktok-row.net）

# from openai import AzureOpenAI

# client = AzureOpenAI(
#     api_key="fCTG9hlWLCmMr7ZfEiWhezW0rCPYo274_GPT_AK",
#     azure_endpoint="https://aidp-i18ntt-sg.byteintl.net/api/modelhub/online/v2/crawl",
#     api_version="2024-03-01-preview",
# )

# response = client.chat.completions.create(
#     model="gpt-5.4-pro-2026-03-05",
#     messages=[
#         {
#             "role": "user",
#             "content": [
#                 {
#                     "type": "text",
#                     "text": "What’s in this image?"
#                 },
#                 # 图片最大支持 5 MB，最多可以传入 10 张图片
#                 {
#                     "type": "image_url",
#                     "image_url": {
#                         # 需要保障 url 外网可访问。
#                         "url": "https://static.wixstatic.com/media/ba2cd3_71d0deba7b87452b85caa20ee07cb1b9~mv2.jpg/v1/fill/w_585,h_405,al_c,q_80,enc_auto/ba2cd3_71d0deba7b87452b85caa20ee07cb1b9~mv2.jpg",
#                         "detail": "auto" # 支持参数 low, high, or auto  参考文档：https://platform.openai.com/docs/guides/vision#low-or-high-fidelity-image-understanding
#                     }
#                 }
#                 #{
#                 #    "type": "image_url",
#                 #    "image_url": {
#                 #        "url": "data:image/jpeg;base64,${base64}", # 如果传入的是 base64 编码，需要为 data URI 的格式。需要带上类似于如下的协议前缀
#                 #        "detail": "auto"
#                 #    }
#                 #}
#             ]
#         }
#     ],
#     max_tokens=500,
#     stream=False,
#     extra_headers={'X-TT-LOGID': '${your_logid}'}
# )

# print(response.model_dump_json(indent=2))




from openai import AzureOpenAI

client = AzureOpenAI(
    api_key="6myxZoOuBuOEyasZ82gEwXd2yWVh8O7K",
    azure_endpoint="https://aidp-i18ntt-sg.byteintl.net/api/modelhub/online/v2/crawl",
    api_version="2024-03-01-preview",
)

response = client.chat.completions.create(
    # model="gpt-5.2-2025-12-11",
    model ="gpt-5.4-2026-03-05",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "What’s in this image?"
                },
                # 图片最大支持 5 MB，最多可以传入 10 张图片
                {
                    "type": "image_url",
                    "image_url": {
                        # 需要保障 url 外网可访问。
                        "url": "https://static.wixstatic.com/media/ba2cd3_71d0deba7b87452b85caa20ee07cb1b9~mv2.jpg/v1/fill/w_585,h_405,al_c,q_80,enc_auto/ba2cd3_71d0deba7b87452b85caa20ee07cb1b9~mv2.jpg",
                        "detail": "auto" # 支持参数 low, high, or auto  参考文档：https://platform.openai.com/docs/guides/vision#low-or-high-fidelity-image-understanding
                    }
                }
                #{
                #    "type": "image_url",
                #    "image_url": {
                #        "url": "data:image/jpeg;base64,${base64}", # 如果传入的是 base64 编码，需要为 data URI 的格式。需要带上类似于如下的协议前缀
                #        "detail": "auto"
                #    }
                #}
            ]
        }
    ],
    max_tokens=500,
    stream=False,
    extra_headers={'X-TT-LOGID': '${your_logid}'}
)

print(response.model_dump_json(indent=2))