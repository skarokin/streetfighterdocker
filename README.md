assuming you have docker installed...
1. clone github repo
2. cd into project
3. run below commands

```
> docker build -t streetfighterai -f Dockerfile .
> docker run -p 8888:8888 streetfighterai
```

4. to open in browser, check your terminal and click the boxed link

![image](https://github.com/user-attachments/assets/4b9c6dc8-4deb-44d8-a76b-745e765c8425)

5. to train a new agent, open `train.ipynb`. to run the models described in our paper, open `run_developed_models.ipynb`
