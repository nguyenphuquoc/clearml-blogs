# The Hero’s Journey to Deep Learning CodeBase
## [Blog I: Once Upon a Repository: How to Write Readable, Maintainable Code with PyTorch](https://medium.com/p/once-upon-a-repository-how-to-write-readable-maintainable-code-with-pytorch-951f03f6a829?source=email-679430f47f06--writer.postDistributed&sk=3a6953df05559b11fbbc35a258e75ec0)

We all aim to write a maintainable and modular codebase that supports the R&D process from research to production. Key to an efficient and successful deep learning project, this is not an easy feat. That is why we decided to write this blog series -- to share our experience from numerous deep learning projects and demonstrate the way to achieve this goal using open source tools.

Our first post in this series is a tutorial on how to leverage the PyTorch ecosystem and Allegro Trains experiments manager to easily write a readable and maintainable computer vision code tailored for your needs. We focus on two packages from the PyTorch ecosystem, Torchvision and Ignite. Torchvision is a popular package consisting of popular datasets wrappers, model architectures, and common image transformations for computer vision. Ignite is a new library that enables simple and clean adding of metrics reports, early-stopping, model checkpointing and other features to your training loop. In this post, we write a codebase that trains and evaluates a Mask-RCNN model on the COCO dataset. We then register the training data (loss, accuracy, etc) to a Pytorch native Tensorboard and use Allegro Trains experiment & autoML manager to manage and track our training experiments. Through these steps, we achieve a seamless, organized, and productive model training flow.