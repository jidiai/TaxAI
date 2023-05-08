import pygame
import sys

pygame.init()

# 窗口大小
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600

# 创建窗口
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("政府和家庭经济模型")

# 颜色定义
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)

# 政府和家庭定义
class Government:
    def __init__(self):
        self.tax_rate = 0.2  # 税率为20%
        self.money = 0      # 初始资金为0

    def collect_taxes(self, households):
        total_income = sum(household.income for household in households)
        taxes_collected = total_income * self.tax_rate
        self.money += taxes_collected

class Household:
    def __init__(self, income):
        self.income = income  # 初始收入
        self.savings = 0     # 初始储蓄为0

    def work(self, firm):
        self.income += firm.produce_goods() * 10

    def save(self, bank):
        amount_to_save = self.income * 0.5   # 储蓄收入的50%
        self.savings += amount_to_save
        bank.deposit(amount_to_save)

    def consume(self, shop):
        amount_to_spend = min(self.income, self.savings)  # 最小值为收入和储蓄的最小值
        self.savings -= amount_to_spend
        self.income -= amount_to_spend
        shop.sell_goods(amount_to_spend)

class Firm:
    def __init__(self):
        self.goods_produced = 0

    def produce_goods(self):
        self.goods_produced += 1
        return self.goods_produced

class Bank:
    def __init__(self):
        self.money = 0

    def deposit(self, amount):
        self.money += amount

class Shop:
    def __init__(self):
        self.goods = 0

    def sell_goods(self, amount):
        self.goods += amount

# 初始化政府、家庭、公司、银行和商店
government = Government()
households = [Household(income=100) for _ in range(100)]
firm = Firm()
bank = Bank()
shop = Shop()

# 渲染循环
while True:
    # 事件处理
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    # 绘制背景
    screen.fill(WHITE)

    # 绘制政府
    government_text = f"政府: {government.money:.2f}元"
    government_font = pygame.font.Font(None, 36)
    government_surface = government_font.render(government_text, True, BLACK)
    government_rect = government_surface.get_rect()
    government_rect.topright = (WINDOW_WIDTH - 10, 10)
    screen.blit(government_surface, government_rect)

    # 绘制家庭
    for i, household in enumerate(households):
        household_text = f"家庭{i}: {household.income:.2f}元 / {household.savings:.2f}元"
        household_font = pygame.font.Font(None, 24)
        household_surface = household_font.render(household_text, True, BLACK)
        household_rect = household_surface.get_rect()
        household_rect.topleft = (10, 10 + i * 30)
        screen.blit(household_surface, household_rect)

    # 绘制公司、银行和商店
    firm_text = f"公司: {firm.goods_produced}件商品"
    firm_font = pygame.font.Font(None, 24)
    firm_surface = firm_font.render(firm_text, True, BLACK)
    firm_rect = firm_surface.get_rect()
    firm_rect.bottomleft = (10, WINDOW_HEIGHT - 10)
    screen.blit(firm_surface, firm_rect)

    bank_text = f"银行: {bank.money:.2f}元"
    bank_font = pygame.font.Font(None, 24)
    bank_surface = bank_font.render(bank_text, True, BLACK)
    bank_rect = bank_surface.get_rect()
    bank_rect.bottomleft = (firm_rect.right + 20, WINDOW_HEIGHT - 10)
    screen.blit(bank_surface, bank_rect)

    shop_text = f"商店: {shop.goods}件商品"
    shop_font = pygame.font.Font(None, 24)
    shop_surface = shop_font.render(shop_text, True, BLACK)
    shop_rect = shop_surface.get_rect()
    shop_rect.bottomleft = (bank_rect.right + 20, WINDOW_HEIGHT - 10)
    screen.blit(shop_surface, shop_rect)

    # 更新屏幕
    pygame.display.update()

