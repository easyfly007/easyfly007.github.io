// @vitest-environment jsdom
import { act, cleanup, fireEvent, render, screen, within } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import App from './App';

function last<T>(list: T[]): T {
    return list[list.length - 1];
}

function click(name: string | RegExp) {
    fireEvent.click(screen.getByRole('button', { name }));
}

describe('App flow', () => {
    beforeEach(() => {
        window.localStorage.clear();
        vi.useFakeTimers();
        vi.spyOn(window, 'prompt').mockImplementation((_m, def) => def ?? '');
        vi.spyOn(window, 'confirm').mockReturnValue(true);
    });

    afterEach(() => {
        cleanup();
        vi.useRealTimers();
        vi.restoreAllMocks();
    });

    it('generates, saves, and plays a workout', () => {
        render(<App />);
        click('哑铃');
        click('生成训练');

        expect(screen.getByText('热身')).toBeTruthy();
        expect(screen.getByText(/^主训练/)).toBeTruthy();
        expect(screen.getByText('放松')).toBeTruthy();
        expect(document.querySelectorAll('.item-detail .body-map').length).toBeGreaterThan(0);

        click('保存为模板');
        expect(JSON.parse(window.localStorage.getItem('gym.saved.v1')!)).toHaveLength(1);

        click('▶ 开始跟练');
        expect(screen.getByText('准备')).toBeTruthy();
        expect(screen.getByText('10')).toBeTruthy();

        act(() => {
            vi.advanceTimersByTime(3000);
        });
        expect(screen.getByText('7')).toBeTruthy();

        click('跳过');
        act(() => {
            vi.advanceTimersByTime(200);
        });
        expect(document.querySelector('.player.work')).toBeTruthy();

        click('暂停');
        const before = document.querySelector('.countdown')!.textContent;
        act(() => {
            vi.advanceTimersByTime(5000);
        });
        expect(document.querySelector('.countdown')!.textContent).toBe(before);
        click('继续');

        click('结束');
        click('修改条件');
        const saved = screen.getByText('我的模板').closest('section')!;
        expect(within(saved).getByText(/^室内-瑜伽垫\+哑铃-30分钟/)).toBeTruthy();
    });

    it('shows a report after finishing and keeps it in the history', () => {
        render(<App />);
        click('15 分钟');
        click('生成训练');
        click('▶ 开始跟练');
        act(() => {
            vi.advanceTimersByTime(20 * 60 * 1000);
        });

        expect(screen.getByText('🎉 训练完成')).toBeTruthy();
        expect(screen.getByText('100%')).toBeTruthy();
        expect(screen.getByText('最近 7 天')).toBeTruthy();
        fireEvent.change(screen.getByLabelText('体重（公斤）'), { target: { value: '60' } });
        click('计算');
        expect(screen.getByText(/^约 \d+ 千卡$/)).toBeTruthy();

        // 第一次训练后提示保存档案；这个浏览器（测试环境）不能直接读写文件，"以后再说"
        expect(screen.getByText('保存训练档案')).toBeTruthy();
        click('以后再说');
        expect(screen.queryByText('保存训练档案')).toBeNull();

        click('返回');
        click('👤 我的');
        const history = screen.getByText('训练记录').closest('section')!;
        expect(within(history).getByText(/^室内-瑜伽垫-15分钟/)).toBeTruthy();
        expect(screen.getByText('训练日历')).toBeTruthy();
        expect(screen.getByText('本月总结')).toBeTruthy();
        expect(JSON.parse(window.localStorage.getItem('gym.history.v1')!)).toHaveLength(1);

        // 填名字
        fireEvent.change(screen.getByLabelText('你的名字'), { target: { value: '小明' } });
        click('保存');
        expect(screen.getByText('小明')).toBeTruthy();
        expect(screen.getByText(/不能直接读写本地文件/)).toBeTruthy();
    });

    it('shows a report when ending early after the first minute', () => {
        render(<App />);
        click('生成训练');
        click('▶ 开始跟练');
        act(() => {
            vi.advanceTimersByTime(90 * 1000);
        });
        click('结束');
        expect(screen.getByText(/^训练报告 · 完成 \d+%$/)).toBeTruthy();
        expect(JSON.parse(window.localStorage.getItem('gym.history.v1')!)).toHaveLength(1);
    });

    it('does not record a workout ended within the first minute', () => {
        render(<App />);
        click('生成训练');
        click('▶ 开始跟练');
        act(() => {
            vi.advanceTimersByTime(5000);
        });
        click('结束');
        expect(screen.getByText('修改条件')).toBeTruthy();
        expect(screen.getByText(/不足 60 秒，没有记录/)).toBeTruthy();
        expect(window.localStorage.getItem('gym.history.v1')).toBeNull();
    });

    it('applies the chosen rest level to the plan', () => {
        render(<App />);
        click('力量');
        click('长一些');
        click('生成训练');
        expect(screen.getByText('组间休息 90 秒')).toBeTruthy();
    });

    it('remembers the coach and voice choice', () => {
        const first = render(<App />);
        click('男教练');
        click('男声');
        first.unmount();

        render(<App />);
        expect(screen.getByRole('button', { name: '男教练' }).getAttribute('aria-pressed')).toBe('true');
        expect(screen.getByRole('button', { name: '男声' }).getAttribute('aria-pressed')).toBe('true');
    });

    it('browses the exercise library, likes an exercise and adds it to the plan', () => {
        render(<App />);
        click('浏览动作库');
        expect(screen.getByText(/^动作库（\d+ 个）$/)).toBeTruthy();
        fireEvent.change(screen.getByLabelText('搜索动作名'), { target: { value: '臀桥' } });
        click('喜欢 臀桥');
        expect(JSON.parse(window.localStorage.getItem('gym.favorites.v1')!)).toEqual(['glute_bridge']);
        expect(screen.queryByRole('button', { name: '加入当前训练 臀桥' })).toBeNull();
        click('返回');

        click('生成训练');
        const before = screen.getAllByRole('button', { name: /^移除 / }).length;
        click('＋ 从动作库添加');
        fireEvent.change(screen.getByLabelText('搜索动作名'), { target: { value: '臀桥' } });
        click('加入当前训练 臀桥');
        expect(screen.getByText('已加入当前训练：臀桥')).toBeTruthy();
        click('返回');
        expect(screen.getAllByRole('button', { name: /^移除 / }).length).toBe(before + 1);
        fireEvent.click(last(screen.getAllByRole('button', { name: '移除 臀桥' })));
        expect(screen.getAllByRole('button', { name: /^移除 / }).length).toBe(before);
    });

    it('speaks short tips during an exercise and supports fullscreen with pause and exit', () => {
        render(<App />);
        click('生成训练');
        click('▶ 开始跟练');
        click('跳过');
        act(() => {
            vi.advanceTimersByTime(5300);
        });
        expect(screen.getByText(/^要点：/)).toBeTruthy();

        click('全屏');
        click('退出全屏');
        expect(screen.queryByRole('button', { name: '退出全屏' })).toBeNull();
        click('全屏');
        const overlay = document.querySelector('.full-overlay') as HTMLElement;
        fireEvent.click(within(overlay).getByRole('button', { name: '暂停' }));
        expect(within(overlay).getByRole('button', { name: '继续' })).toBeTruthy();
        fireEvent.click(within(overlay).getByRole('button', { name: '退出全屏' }));
        expect(document.querySelector('.full-overlay')).toBeNull();
    });

    it('restores the last conditions on reload', () => {
        const first = render(<App />);
        click('杠铃');
        click('生成训练');
        first.unmount();

        render(<App />);
        expect(screen.getByRole('button', { name: '杠铃' }).getAttribute('aria-pressed')).toBe('true');
    });
});
