import torch
from tqdm import tqdm




def train_transformer(model, criterion, optimizer, scheduler, dataloader, eval_dataloader, num_epoch, device=torch.device('cuda')):
    avg_losses_train = []
    avg_losses_val = []
    best_val_loss = float('inf')

    for epoch in range(1, num_epoch + 1):
        print(f'Epoch: {epoch}')
        model.train()
        train_losses = []
        for source, target in tqdm(dataloader):
            optimizer.zero_grad()
            
            # Подготовка данных
            source, target_input = source.to(device), target[:, :-1].to(device)
            target_output = target[:, 1:].to(device).flatten()

            # Прямой проход
            output = model(source, target_input)
            output = output.view(-1, output.size(-1))
            
            
            # print(f"Logits max: {output.max().item()}, min: {output.min().item()}")

            # Вычисление ошибки
            loss = criterion(output, target_output)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        avg_train_loss = sum(train_losses) / len(train_losses)
        avg_losses_train.append(avg_train_loss)
        print(f'Average train loss: {avg_train_loss:.4f}')
        
        # Оценка на валидации
        model.eval()
        val_losses = []

        with torch.no_grad():
            for source, target in tqdm(eval_dataloader):
                source, target_input = source.to(device), target[:, :-1].to(device)
                target_output = target[:, 1:].to(device).flatten()

                
                output = model(source, target_input)
                output = output.view(-1, output.size(-1))
                loss = criterion(output, target_output)
                val_losses.append(loss.item())

        avg_val_loss = sum(val_losses) / len(val_losses)
        avg_losses_val.append(avg_val_loss)
        print(f'Average val loss: {avg_val_loss:.4f}')
        scheduler.step(avg_val_loss)
        # Сохранение лучшей модели
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            print('Model saved.')

    return avg_losses_train, avg_losses_val